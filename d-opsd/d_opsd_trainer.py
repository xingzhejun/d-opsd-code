import torch
from trl.trainer.grpo_trainer import GRPOTrainer
from typing import Any, Callable, Optional, Union, Sized
import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase, TrainerCallback, Trainer
from datasets import Dataset, IterableDataset
import warnings
import torch.nn.functional as F
from trl.trainer.grpo_config import GRPOConfig
from trl.extras.profiling import profiling_decorator, profiling_context
from transformers.utils import is_peft_available
from torch import nn
from trl.import_utils import is_rich_available, is_vllm_available
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import (
    generate_model_card,
    get_comet_experiment_url,
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)

from utils import main_print, generate, get_all_parsed_answer, get_parsed_answer_sudoku, get_parsed_answer_countdown


if is_peft_available():
    from peft import PeftConfig, get_peft_model
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class dOPSDTrainer(GRPOTrainer):
    """
    On-policy Self-distillation (OPSD) Trainer for Diffusion Language Models.

    This class extends from the GRPOTrainer. Very Important: Make Sure You Have Replaced the trl File with Ours.

    Key features:
    - Learn from the self-generated future: retain a part of the teacher's trajectory to provide a learning signal.
    - Efficient per-step divergence supervision for diffusion language models
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None,
            None,
        ),
        peft_config: Optional["PeftConfig"] = None,
    ):
        # Initialize the parent class
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

        self.log_completions = True
        self.batch_divide = args.batch_divide
        self.debug1 = args.debug1 # cnannot replace with debug, there is already one "debug" attribute existed.
        self.passk = args.passk
        self.passk_temperature = args.passk_temperature
        self.teacher_retain_ratio = args.teacher_retain_ratio
        self.fixed_teacher = args.fixed_teacher
        self.top_k_loss = args.top_k_loss
        self.jsd_token_clip = args.jsd_token_clip
        self.add_ref = args.add_ref
        self.diff_student_mask = args.diff_student_mask
        self.dataset_name = args.dataset
        self.sudoku_threshold = args.sudoku_threshold
        if self.add_ref:
            self.teacher_max_prompt_length = args.teacher_max_prompt_length
        if args.max_grad_norm is not None:
            main_print(f'max_grad is {args.max_grad_norm}')
        else:
            main_print(f'no max_grad')
        main_print(f"Batch divide to prevent OOM: {self.batch_divide}")
        main_print(f"Debug mode: {self.debug1}")
        main_print(f"PassK (number of reasoning trajectories): {self.passk}")
        main_print(f"PassK temperature: {self.passk_temperature}")
        main_print(f"Teacher retain ratio: {self.teacher_retain_ratio}")
        main_print(f"Fixed teacher: {self.fixed_teacher}")
        main_print(f"Top-k for loss computation: {self.top_k_loss}")
        main_print(f"JSD token clip value: {self.jsd_token_clip}")
        main_print(f"Add reference solutions to prompts: {self.add_ref}")
        main_print(f"Diff student mask: {self.diff_student_mask}")
        main_print(f"Dataset name: {self.dataset_name}")
        main_print(f"Sudoku accuracy threshold: {self.sudoku_threshold}")
        main_print(f'gen_length: {self.args.max_completion_length}, block_length: {self.args.block_length}, diffusion_steps: {self.args.diffusion_steps}')

    def get_logits(self, model, batch, prompt_index, cfg_scale, mask_id):
        input = batch
        logits = model(input).logits
        if cfg_scale > 0.0:
            main_print(f'cfg>0, Wrong')
            raise NotImplementedError("CFG is not implemented for dOPSDTrainer yet")
        return logits
    
    def generalized_jsd_loss(
        self,
        student_logits,
        teacher_logits,
        beta=0.5,
        reduction="batchmean",
        top_k=None,
        token_clip=None,
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')
            top_k:
                If set, restricts the loss to only the top-k tokens of the teacher distribution. Both student and
                teacher distributions are renormalized over these k tokens before computing JSD. This reduces memory
                and focuses distillation on the teacher's most probable tokens. (default: None = full vocabulary)
            token_clip:
                if set, clips per-token divergence values to this maximum before reduction. Prevents style tokens from dominating the gradient signal over math tokens.

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        if top_k is not None and top_k > 0:
            # Restrict to top-k tokens of the teacher distribution and renormalize.
            # Also compute the overlap between student top-k and teacher top-k.
            # Shape: [batch, seq_len, top_k]
            _, teacher_top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
            _, student_top_k_indices = torch.topk(student_logits, k=top_k, dim=-1)

            student_top_k_mask = student_top_k_indices.unsqueeze(-1) == teacher_top_k_indices.unsqueeze(-2)
            top_k_overlap = student_top_k_mask.any(dim=-1).float().mean(dim=-1)

            student_logits = torch.gather(student_logits, dim=-1, index=teacher_top_k_indices)
            teacher_logits = torch.gather(teacher_logits, dim=-1, index=teacher_top_k_indices)
        else:
            top_k_overlap = torch.zeros(
                student_logits.shape[:-1], dtype=student_logits.dtype, device=student_logits.device
            )

        # Compute log probabilities for student and probabilities for teacher
        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0: # forward
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1: # reverse
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )

            # Compute KL divergences using F.kl_div
            # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)
            # Compute the Generalized Jensen-Shannon Divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Per-token clipping: cap each token's divergence value. This refers to the pointwise KL clipping in the paper.
        if token_clip is not None:
            clipped_mask = jsd > token_clip
            clip_ratio = clipped_mask.float().mean()
            jsd = jsd.clamp(max=token_clip)
        else:
            clip_ratio = torch.zeros((), dtype=jsd.dtype, device=jsd.device)

        # Apply reduction
        if reduction == "batchmean":
            loss = jsd.sum() / jsd.size(0)
        elif reduction == "sum":
            loss = jsd.sum()
        elif reduction == "mean":
            loss = jsd.mean()
        else:
            loss = jsd

        return loss, clip_ratio, top_k_overlap
    
    @profiling_decorator
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        '''
        inputs {
            "prompt_length": prompt_length,
            "teacher_prompt_length": teacher_prompt_length,
            "trajectory": trajectory.cpu(),
            "teacher_trajectory": teacher_trajectory.cpu(),
            "steps": steps, 
            "gen_length": gen_length,
            "block_length": block_length,
            "is_correct": is_correct,
            }
        '''
        
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        prompt_length, teacher_prompt_length, steps, gen_length, block_length = inputs["prompt_length"], inputs["teacher_prompt_length"], inputs["steps"], inputs["gen_length"], inputs["block_length"]
        is_correct = inputs["is_correct"]
        bsz_per_step = (inputs["trajectory"].shape[0] - 1) // self.batch_divide # to prevent OOM
        if bsz_per_step == 0:
            raise ValueError(f"Batch size per step is zero. Check if the batch_divide {self.batch_divide} is too large for the current batch size {inputs['trajectory'].shape[0] - 1}.")
        start_pos = ((self._step - 1) % self.batch_divide) * bsz_per_step
        end_pos = start_pos + bsz_per_step
        student_input = inputs["trajectory"][start_pos:end_pos].to(model.device)
        student_output = inputs["trajectory"][start_pos + 1:end_pos + 1].to(model.device)
        teacher_input = inputs["teacher_trajectory"][start_pos:end_pos].to(model.device)
        
        # teacher forward
        if self.fixed_teacher and is_peft_model(model):
            with torch.no_grad(), self.accelerator.unwrap_model(model).disable_adapter():
                teacher_logits = self.get_logits(model, teacher_input, None, self.args.cfg_scale, self.args.mask_id)
        else:
            with torch.no_grad():
                teacher_logits = self.get_logits(model, teacher_input, None, self.args.cfg_scale, self.args.mask_id)
        teacher_logits = teacher_logits.detach()
        
        if (not self.add_ref) and self.diff_student_mask:
            diff_mask = student_input != student_output  # [bsz_per_step, seq_length]. This identifies the computation positions.
        else:
            mask_id = self.args.mask_id
            seq_length = teacher_input.size(1)
            # Refer to the "selection from the teacher distribution" in the paper: Use teacher confidence to pick 2 masked positions in the current block.
            teacher_confidence = teacher_logits.max(dim=-1).values  # [bsz_per_step, seq_length]
            diff_mask = torch.zeros_like(teacher_input, dtype=torch.bool)

            for i in range(teacher_input.size(0)):
                masked_positions = torch.where(teacher_input[i] == mask_id)[0]
                if masked_positions.numel() == 0:
                    raise ValueError(
                        f"No mask_id found in teacher_input at row {i}. Cannot build diff_mask."
                    )

                first_mask_pos = masked_positions[0]
                relative_pos = first_mask_pos - teacher_prompt_length
                if relative_pos < 0:
                    raise ValueError(
                        f"First mask position ({first_mask_pos.item()}) is before teacher prompt_length ({teacher_prompt_length}) at row {i}."
                    )

                block_idx = relative_pos // block_length
                block_start = teacher_prompt_length + block_idx * block_length
                block_end = min(block_start + block_length, seq_length)

                block_positions = torch.arange(block_start, block_end, device=teacher_input.device)
                block_masked_positions = block_positions[teacher_input[i, block_positions] == mask_id]

                if block_masked_positions.numel() < 2:
                    raise ValueError(
                        f"Expected at least 2 mask_id positions in block [{block_start}, {block_end}) for row {i}, "
                        f"but found {block_masked_positions.numel()}."
                    )

                block_confidence = teacher_confidence[i, block_masked_positions]
                top2_relative = torch.topk(block_confidence, k=2, dim=0).indices
                top2_positions = block_masked_positions[top2_relative]
                diff_mask[i, top2_positions] = True
        diff_counts = diff_mask.sum(dim=1)
        if not torch.all(diff_counts == 2): 
            # Here we fix it as 2 because all our trainings are based on decoding 2 toekns at each step. You can adjust it to yours.
            bad_rows = torch.nonzero(diff_counts != 2, as_tuple=True)[0]
            raise ValueError(
                f"Each student_input/student_output pair must differ at exactly 2 positions, "
                f"but got counts={diff_counts.tolist()}, bad_rows={bad_rows.tolist()}."
                f" "
            )
        teacher_idx_selection = torch.stack([torch.where(diff_mask[i])[0] for i in range(diff_mask.size(0))], dim=0) # [bsz_per_step, 2]
        if self.add_ref:
            assert teacher_prompt_length != prompt_length
            idx_selection = teacher_idx_selection - (teacher_prompt_length - prompt_length)
        else:
            assert teacher_prompt_length == prompt_length
            idx_selection = teacher_idx_selection
        if self.debug1:
            main_print(f'step is:{self._step}')
            main_print(f'global step is: {self.state.global_step}')
            main_print(f'start_pos: {start_pos}, end_pos: {end_pos}')
            main_print(f'trajectory shape: {student_input.shape}')
            main_print(f'idx_selection shape: {idx_selection.shape}')
            main_print(f'idx_selection[5] is: {idx_selection[5]}')
            
        # student forward
        student_logits = self.get_logits(model, student_input, None, self.args.cfg_scale, self.args.mask_id)
        if self.debug1:
            main_print(f'Before logits cutting')
            main_print(f'student_logits shape: {student_logits.shape}')
            main_print(f'teacher_logits shape: {teacher_logits.shape}')

        idx_selection_expanded = idx_selection.unsqueeze(-1).expand(-1, -1, student_logits.size(-1))
        teacher_idx_selection_expanded = teacher_idx_selection.unsqueeze(-1).expand(-1, -1, teacher_logits.size(-1))
        student_logits = torch.gather(student_logits, dim=1, index=idx_selection_expanded)
        teacher_logits = torch.gather(teacher_logits, dim=1, index=teacher_idx_selection_expanded)
        if self.debug1:
            main_print(f'After logits cutting')
            main_print(f'student_logits shape: {student_logits.shape}')
            main_print(f'teacher_logits shape: {teacher_logits.shape}')

        loss, clip_ratio, top_k_overlap = self.generalized_jsd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                beta=self.beta,
                top_k=self.top_k_loss,
                token_clip=self.jsd_token_clip,
            )   
        if self.debug1:
            main_print(f'After generalized_jsd_loss: loss={loss.item():.6f}, clip_ratio={clip_ratio.item():.6f}, top_k_overlap_shape={tuple(top_k_overlap.shape)}')
        assert top_k_overlap.shape == (student_logits.size(0), idx_selection.size(1))
        
        mode = "eval" if self.control.should_evaluate else "train"
        self._metrics[mode]["loss"].append(self.accelerator.gather_for_metrics(loss).mean().item())
        self._metrics[mode]["clip_ratio"].append(
            self.accelerator.gather_for_metrics(clip_ratio).mean().item()
        )
        # top_k_overlap has shape [local_batch, local_seq_len], which may vary across ranks.
        # Reduce locally to a scalar first to avoid distributed gather shape mismatch / deadlock.
        top_k_overlap_local = top_k_overlap.mean()
        top_k_overlap_value = self.accelerator.gather_for_metrics(top_k_overlap_local).mean().item()
        self._metrics[mode]["top_k_overlap"].append(top_k_overlap_value)
        if self.debug1:
            main_print(f'After metrics gather: top_k_overlap_value={top_k_overlap_value:.6f}')

        del student_logits, teacher_logits, diff_mask, student_input, student_output, teacher_input
        torch.cuda.empty_cache()
        
        # Refer to the "Computeonly on Correct Generations" in the paper. For ablations, just replace followings with "return loss".
        if self.dataset_name == "sudoku":
            if is_correct >= self.sudoku_threshold:
                return loss
            else:
                return loss * 0.0
        # return loss
        elif is_correct or self.add_ref:
            return loss
        else:
            return loss * 0.0

    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        mode = "eval" if self.control.should_evaluate else "train"

        if mode == "train":
            # Very important, due to the RepeatSampler from GPPOTrainer.
            if self._step % self.batch_divide == 0:
                inputs = self._generate_and_score_completions(inputs)
                self._buffered_inputs[0] = inputs
            else:
                inputs = self._buffered_inputs[0]
            self._step += 1
        else:
            inputs = self._generate_and_score_completions(inputs)
        return inputs

    def _generate_and_score_completions(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs
        ]
        if self.debug1:
            main_print(f'prompts_text is: {prompts_text}')
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids = prompt_inputs["input_ids"]
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        if self.debug1:
            main_print(f'prompt_ids shape is: {prompt_ids.shape}')  

        # Configuration for the diffusion generation
        gen_length = self.args.max_completion_length
        block_length = self.args.block_length
        steps = self.args.diffusion_steps
        temperature = self.args.temperature or 0.0
        cfg_scale = self.args.cfg_scale

        with unwrap_model_for_generation(self.model_wrapped, self.accelerator) as unwrapped_model:
            generation_batch_size = 1 # we fix it here. It almost won't slow the training..
            with torch.no_grad():
                for i in range(0, prompt_ids.size(0), generation_batch_size):
                    end_idx = min(i + generation_batch_size, prompt_ids.size(0))
                    batch_prompt_ids = prompt_ids[i:end_idx] # [1, prompt_length]
                    # WARNING: Attention masks are not currently used during generation.
                    # This works fine here as long as the generation batch only consists of same prompts (our case a single prompt).
                    
                    batch_prompt_completion_ids, batch_trajectory = generate(
                        model=unwrapped_model,
                        prompt=batch_prompt_ids,
                        steps=steps,
                        gen_length=gen_length,
                        block_length=block_length,
                        temperature=temperature,
                        cfg_scale=cfg_scale,
                        remasking=self.args.remasking,
                        mask_id=self.args.mask_id,
                        debug1=self.debug1,
                        fp16=self.args.fp16
                    )
                    completions_text = self.processing_class.batch_decode(batch_prompt_completion_ids[:, -gen_length:], skip_special_tokens=False)
                    # if self.debug1:
                    #     main_print(f'prompt type: {type(inputs[0]["prompt"])}') # list
                    if self.dataset_name == "sudoku":
                        parsed_answer, accuracy = get_parsed_answer_sudoku(completions_text[0], inputs[0]["solution"], inputs[0]["puzzle"])
                        best_accuracy = accuracy
                        best_parsed_answer = parsed_answer
                        best_completion_text = completions_text
                        best_batch_prompt_completion_ids = batch_prompt_completion_ids
                        best_batch_trajectory = batch_trajectory
                        is_correct = False
                    elif self.dataset_name == "countdown":
                        parsed_answer, is_correct = get_parsed_answer_countdown(completions_text[0], inputs[0]["numbers"], inputs[0]["target"])
                    else:
                        parsed_answer, is_correct = get_all_parsed_answer(completions_text[0], inputs[0]["answer"], self.dataset_name)
                    if self.debug1:
                        main_print(f'input is:{inputs}')  
                        '''
                        [
                            {
                                "question": "",
                                "answer": "",
                                "prompt": [ {"role": "user", "content": "prompt+question"} ]
                            }
                        ]
                        '''
                        main_print(f'completions_text is: {completions_text[0]}')
                        main_print(f'parsed_answer is: {parsed_answer}')
                        if self.dataset_name == "sudoku":
                            gt_answer = inputs[0]["solution"]
                        elif self.dataset_name == "countdown":
                            gt_answer = inputs[0]["target"]
                        else:
                            gt_answer = inputs[0]["answer"]
                        main_print(f'ground truth answer is: {gt_answer}')
                        main_print(f'is_correct is: {is_correct if self.dataset_name != "sudoku" else accuracy}')
                    
                    # Refer to the "pass@k" in the paper: extend to more reasoning trajectories if needed.
                    iter_num = 1
                    while (not self.add_ref) and iter_num < self.passk and (not is_correct or self.dataset_name == "sudoku"):
                        iter_num = iter_num + 1
                        batch_prompt_completion_ids, batch_trajectory = generate(
                            model=unwrapped_model,
                            prompt=batch_prompt_ids,
                            steps=steps,
                            gen_length=gen_length,
                            block_length=block_length,
                            temperature=self.passk_temperature,
                            cfg_scale=cfg_scale,
                            remasking=self.args.remasking,
                            mask_id=self.args.mask_id,
                            debug1=self.debug1,
                            fp16=self.args.fp16
                        )
                        completions_text = self.processing_class.batch_decode(batch_prompt_completion_ids[:, -gen_length:], skip_special_tokens=False)
                        if self.dataset_name == "sudoku":
                            parsed_answer, accuracy = get_parsed_answer_sudoku(completions_text[0], inputs[0]["solution"], inputs[0]["puzzle"])
                            if accuracy >= best_accuracy:
                                best_accuracy = accuracy
                                best_parsed_answer = parsed_answer
                                best_completion_text = completions_text
                                best_batch_prompt_completion_ids = batch_prompt_completion_ids
                                best_batch_trajectory = batch_trajectory
                        elif self.dataset_name == "countdown":
                            parsed_answer, is_correct = get_parsed_answer_countdown(completions_text[0], inputs[0]["numbers"], inputs[0]["target"])
                        else:
                            parsed_answer, is_correct = get_all_parsed_answer(completions_text[0], inputs[0]["answer"], self.dataset_name)
                        if self.debug1:
                            main_print(f'now at iteration: {iter_num}')
                            main_print(f'completions_text is: {completions_text[0]}')
                            main_print(f'parsed_answer is: {parsed_answer}')
                            if self.dataset_name == "sudoku":
                                gt_answer = inputs[0]["solution"]
                            elif self.dataset_name == "countdown":
                                gt_answer = inputs[0]["target"]
                            else:
                                gt_answer = inputs[0]["answer"]
                            main_print(f'ground truth answer is: {gt_answer}')
                            main_print(f'is_correct is: {is_correct if self.dataset_name != "sudoku" else best_accuracy}')    
                    '''
                    batch_prompt_completion_ids: [1, prompt_length + gen_length]
                    batch_trajectory: [x0, x1, ..., x_steps_till_eos], each of shape [1, prompt_length + gen_length]
                    '''
            # The correct here is for pass@k. If correct, we only keep the first succesful trajectory; Otherwise we keep the last sampled trajectory.
            local_is_correct = torch.tensor(iter_num, device=device, dtype=torch.float32)

        if self.dataset_name == "sudoku":
            accuracy = best_accuracy
            parsed_answer = best_parsed_answer
            completions_text = best_completion_text
            batch_prompt_completion_ids = best_batch_prompt_completion_ids
            batch_trajectory = best_batch_trajectory
            accuracy_tensor = torch.tensor(accuracy, device=device, dtype=torch.float32)
        prompt_length = prompt_ids.size(1)
        completion_part_ids = batch_prompt_completion_ids[:, prompt_length:]
        eos_id = 126081 # we fix it here for LLADA. Please adjust it to yours.
        eos_positions = torch.where(completion_part_ids[0] == eos_id)[0]
        pure_gen_length_val = eos_positions[0].item() if eos_positions.numel() > 0 else completion_part_ids.size(1)
        pure_gen_length = torch.tensor(pure_gen_length_val, device=device, dtype=torch.float32)
        trajectory = torch.cat(batch_trajectory, dim=0) # [steps_till_eos, length], student

        steps_till_eos, full_seq_length = trajectory.shape
        if steps_till_eos <= self.batch_divide:
            print(f'completions_text is: {completions_text[0]}')
            raise ValueError(f"Steps till EOS {steps_till_eos} is not greater than batch_divide {self.batch_divide}, which may cause issues with batch splitting. Consider reducing batch_divide or checking the generation process for early EOS.")

        # construct the teacher
        teacher_trajectory = trajectory.clone()
        if self.add_ref: # AR-style counterpart that adds reference solutions to the prompt
            teacher_prompts_text = [
                maybe_apply_chat_template(example, self.processing_class)["teacher_prompt"] for example in inputs
            ]
            if self.debug1:
                main_print(f'teacher_prompts_text is: {teacher_prompts_text}')
            teacher_prompt_inputs = self.processing_class(
                text=teacher_prompts_text,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
            teacher_prompt_inputs = Trainer._prepare_inputs(self, teacher_prompt_inputs)
            teacher_prompt_ids = teacher_prompt_inputs["input_ids"]
            if self.teacher_max_prompt_length is not None:
                teacher_prompt_ids = teacher_prompt_ids[:, -self.teacher_max_prompt_length:]

            teacher_prompt_length = teacher_prompt_ids.size(1)
            teacher_prompt_ids = teacher_prompt_ids.to(teacher_trajectory.device)
            teacher_prompt_ids = teacher_prompt_ids[0:1]
            teacher_completion_part = teacher_trajectory[:, prompt_length:]
            teacher_prompt_part = teacher_prompt_ids.expand(steps_till_eos, -1)
            teacher_trajectory = torch.cat([teacher_prompt_part, teacher_completion_part], dim=1)
        else:
            teacher_prompt_length = prompt_length
            final_sequence = batch_prompt_completion_ids[0]  # [seq_length]
            
            for step_idx in range(steps_till_eos - 1):
                n = step_idx // (block_length // (gen_length // steps))
                start_pos = prompt_length + (n + 1) * block_length
                if start_pos > full_seq_length - block_length:
                    raise ValueError(f"start_pos {start_pos} for step {step_idx} is out of bounds for full_seq_length {full_seq_length}")

                # Skip if start_pos is already at or beyond the EOS position
                # print(f'step_idx: {step_idx}, n: {n}, start_pos: {start_pos}, pure_gen_length_val: {pure_gen_length_val}, prompt_length: {prompt_length}')
                if start_pos >= pure_gen_length_val + prompt_length:
                    continue

                # candidate_positions = torch.arange(start_pos, pure_gen_length_val + prompt_length, device=teacher_trajectory.device)
                candidate_positions = torch.arange(start_pos, full_seq_length, device=teacher_trajectory.device)
                num_candidates = candidate_positions.numel()
                num_replace = int(num_candidates * self.teacher_retain_ratio)
                if num_replace <= 0:
                    continue
                selected_relative = torch.randperm(num_candidates, device=teacher_trajectory.device)[:num_replace]
                selected_positions = candidate_positions[selected_relative]
                teacher_trajectory[step_idx, selected_positions] = final_sequence[selected_positions]
        
        if self.debug1:
            main_print(f'trajectory shape: {trajectory.shape}')
            main_print(f'teacher_trajectory shape: {teacher_trajectory.shape}')
            main_print(f"Prompt length: {prompt_length}")
            completions_text_10step = self.processing_class.batch_decode(trajectory[10:11, -gen_length:], skip_special_tokens=False)
            teacher_completions_text_10step = self.processing_class.batch_decode(teacher_trajectory[10:11, -gen_length:], skip_special_tokens=False)
            main_print(f'trajectory[10]: {completions_text_10step[0]}')
            main_print(f'teacher_trajectory[10]: {teacher_completions_text_10step[0]}')
        

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"
        completion_length = self.accelerator.gather_for_metrics(pure_gen_length).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)
        mean_is_correct = self.accelerator.gather_for_metrics(local_is_correct).mean().item()
        self._metrics[mode]["iter_num"].append(mean_is_correct)
        if self.dataset_name == "sudoku":
            if accuracy >= self.sudoku_threshold:
                effective_num = 1
            else:
                effective_num = 0
            accuracy_value = self.accelerator.gather_for_metrics(accuracy_tensor).mean().item()
            effective_num_gathered = self.accelerator.gather_for_metrics(
                torch.tensor(effective_num, device=device, dtype=torch.float32)
            ).mean().item()
            self._metrics[mode]["accuracy"].append(accuracy_value)
            self._metrics[mode]["effective_num"].append(effective_num_gathered)
            is_correct = accuracy

        if self.log_completions and self.state.global_step % self.args.completion_logging_steps == 0:
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            if self.dataset_name == "sudoku":
                rewards_to_log = [accuracy]
            elif is_correct:
                rewards_to_log = [1.0]
            else:
                rewards_to_log = [0.0]
            rewards_to_log = gather_object(rewards_to_log)
            if self.add_ref:
                teacher_prompts_to_log = gather_object(teacher_prompts_text)
                
            if self.accelerator.is_main_process:
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self._step,
                        teacher_prompts_to_log if self.add_ref else None,
                    )

        return {
            "prompt_length": prompt_length,
            "teacher_prompt_length": teacher_prompt_length,
            "trajectory": trajectory.cpu(),
            "teacher_trajectory": teacher_trajectory.cpu(),
            "steps": steps, 
            "gen_length": gen_length,
            "block_length": block_length,
            "is_correct": is_correct,
        }