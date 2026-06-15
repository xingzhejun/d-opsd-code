import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

from generate import generate
from gsm8k import GSM8KDataset
from math500 import MATH500Dataset
from countdown import CTDDataset
from sudoku import SudokuDataset

from utils import main_print, get_parsed_answer, get_parsed_answer_math, get_parsed_answer_sudoku, get_parsed_answer_countdown
from parser_helper import is_equiv


DATASET_MAP = {
    "gsm": GSM8KDataset,
    "math": MATH500Dataset,
    "countdown": CTDDataset,
    "sudoku": SudokuDataset,
}


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_ddp():
    dist.init_process_group("nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def get_all_parsed_answer(generation, answer, dataset, question):
    if dataset == "gsm":
        parsed_answer, effective_tokens = get_parsed_answer(generation, answer) 
        is_correct = parsed_answer is not None and parsed_answer == answer
    elif dataset == "math":
        parsed_answer, effective_tokens = get_parsed_answer_math(generation, answer)
        is_correct = False
        if parsed_answer is not None:
            is_correct = is_equiv(parsed_answer, answer)
    elif dataset == "countdown":
        equation, result, target, effective_tokens, is_valid = get_parsed_answer_countdown(generation, answer, question)
        is_correct = False
        if is_valid:
            if target is not None and abs(result - target) < 1e-5:
                is_correct = True
        parsed_answer = equation
    return parsed_answer, effective_tokens, is_correct


def evaluate(
    model,
    tokenizer,
    dataloader,
    gen_length=128,
    temperature=0.0,
    cfg_scale=0.0,
    steps=64,
    block_length=32,
    calculate_distance=False,
    distance_dir="",
    calculate_confidence=False,
    confidence_dir="",
    remasking="low_confidence",
    num_answer_per_question=1,
    debug=False,
    record_all=False,
    passk=False,
    max_iter_num=8,
    answer_path=None,
    pre_answer_keep_mode="prefix",
    clean_answer=False,
    add_ref=False,
    dataset="gsm",
):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    device = model.device
    batch_idx = 0
    if answer_path:
        with open(answer_path, 'r', encoding='utf-8') as f:
            answer_data = json.load(f)
        answer_data = answer_data["generations"]

    for batch in tqdm(dataloader, disable=(dist.get_rank() != 0)):
        start_time = time.time()
        input_ids = batch["input_ids"].to(device)
        gt_answers = batch["answers"]
        questions = batch["questions"]
        prompts = batch["prompts"]
        '''
        For Countdown dataset:
        gt_answers: (numbers, target)
        questions:
            "Numbers: [44, 19, 35]
            Target: 98"
        '''

        # answer_path refers to Table 3 in the paper.
        if dataset == "sudoku" and passk and input_ids.size(0) == 1:
            repeat_times = 8
            input_ids = input_ids.repeat(repeat_times, 1)

        for _ in range(num_answer_per_question):
            if answer_path:
                pre_answer = answer_data[batch_idx]["generations"]
            else:
                pre_answer = None
            out, distances_list, confidence_list = generate(
                model,
                input_ids,
                tokenizer,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature if (dataset != "sudoku" or not passk) else 1.0,
                cfg_scale=cfg_scale,
                remasking=remasking,
                calculate_distance=calculate_distance,
                calculate_confidence=calculate_confidence,
                pre_answer=pre_answer,
                pre_answer_keep_mode=pre_answer_keep_mode,
                clean_answer=clean_answer,
                debug=debug,
            )
            generated_texts = tokenizer.batch_decode(out[:, -gen_length:], skip_special_tokens=False)
            if dataset == "sudoku":
                if not passk:
                    parsed_answer = None
                    is_correct = None
                else:
                    best_accuracy = 0
                    best_parsed_answer = None
                    best_generated_text = None
                    for bsz_num in range (input_ids.size(0)):
                        parsed_answer, accuracy, correct_cells, empty_cells, effective_tokens = get_parsed_answer_sudoku(generated_texts[bsz_num], gt_answers[0], questions[0])
                        if debug:
                            print(f'accuracy: {accuracy}')
                        if accuracy >= best_accuracy:
                            best_accuracy = accuracy
                            best_parsed_answer = parsed_answer
                            best_generated_text = generated_texts[bsz_num:bsz_num+1]
                    is_correct = best_accuracy
                    parsed_answer = best_parsed_answer
                    generated_texts = best_generated_text
                if debug:
                    print(f'Parsed answer: {parsed_answer}, Accuracy: {is_correct}, ground truth: {gt_answers[0]}')    
            else:
                parsed_answer, effective_tokens, is_correct = get_all_parsed_answer(generated_texts[0], gt_answers[0], dataset, questions[0]) 
            if (not passk) or dataset == "sudoku":
                example_result = [
                    {
                        "question": questions[j],
                        "prompt_input": prompts[j],
                        "generations": generated_texts[j],
                        "parsed_answer": parsed_answer,
                        "ground_truth": gt_answers[j],
                        "is_correct": is_correct,
                    }
                    for j in range(len(gt_answers))
                ]
                all_generations.extend(example_result)
                total_processed += len(generated_texts)
            # For the reported pass@1 accuracy, the branch already ends here.
            
            else:
                if record_all:
                    example_result = [
                        {
                            "question": questions[j],
                            "prompt_input": prompts[j],
                            "generations": generated_texts[j],
                            "parsed_answer": parsed_answer,
                            "ground_truth": gt_answers[j],
                            "is_correct": is_correct,
                        }
                        for j in range(len(gt_answers))
                    ]
                    all_generations.extend(example_result)
                    total_processed += len(generated_texts)

                iter_num = 1
                if debug:
                    print(f'parsed_answer: {parsed_answer}, gt_answer: {gt_answers[0]}, is_correct: {is_correct}')
                while not is_correct:
                    if debug:
                        print(f'Iter num: {iter_num}')
                    if iter_num >= max_iter_num:
                        break
                    out, distances_list, confidence_list = generate(
                        model,
                        input_ids,
                        tokenizer,
                        steps=steps,
                        gen_length=gen_length,
                        block_length=block_length,
                        temperature=1.0,
                        cfg_scale=cfg_scale,
                        remasking=remasking,
                        calculate_distance=calculate_distance,
                        calculate_confidence=calculate_confidence,
                        )

                    generated_texts = tokenizer.batch_decode(out[:, -gen_length:], skip_special_tokens=False)
                    parsed_answer, effective_tokens, is_correct = get_all_parsed_answer(generated_texts[0], gt_answers[0], dataset, questions[0]) 
                    iter_num = iter_num + 1
                    if debug:
                        print(f'parsed_answer: {parsed_answer}, gt_answer: {gt_answers[0]}, is_correct: {is_correct}')

                    if record_all:
                        example_result = [
                            {
                                "question": questions[j],
                                "prompt_input": prompts[j],
                                "generations": generated_texts[j],
                                "parsed_answer": parsed_answer,
                                "ground_truth": gt_answers[j],
                                "is_correct": is_correct,
                            }
                            for j in range(len(gt_answers))
                        ]
                        all_generations.extend(example_result)
                        total_processed += len(generated_texts)
                
                if not record_all:
                    example_result = [
                        {
                            "question": questions[j],
                            "prompt_input": prompts[j],
                            "generations": generated_texts[j],
                            "parsed_answer": parsed_answer,
                            "ground_truth": gt_answers[j],
                            "is_correct": is_correct,
                        }
                        for j in range(len(gt_answers))
                    ]
                    all_generations.extend(example_result)
                    total_processed += len(generated_texts)

        if calculate_distance:
            distance_filename = f"{distance_dir}/batch_{batch_idx}.json"
            with open(distance_filename, "w") as f:
                json.dump(distances_list, f, indent=2)
        if calculate_confidence:
            confidence_filename = f"{confidence_dir}/batch_{batch_idx}.json"
            with open(confidence_filename, "w") as f:
                json.dump(confidence_list, f, indent=2)

        wall_times.append(time.time() - start_time)
        batch_idx += 1

        # Print individual results
        if dist.get_rank() == 0:
            idx = random.randint(0, len(questions) - 1)
            print(f"Question: {questions[idx]}")
            print("-" * 50)
            if add_ref:
                print(f'prompt with reference:\n{prompts[idx]}')
                print("-" * 50)
            print("Generation:")
            print(generated_texts[idx])
            print("-" * 50)
            print(f"Ground truth: {gt_answers[idx]}")
            print("=" * 50)
            print(f'Time for batch: {wall_times[-1]/60:.2f} minutes')
            # if calculate_distance:
            #     print(f'Distances saved to {distance_filename}')
            # if calculate_confidence:
            #     print(f'Confidences saved to {confidence_filename}')
        
        # if debug:
        #     if batch_idx == 10:
        #         break

    avg_wall_time = sum(wall_times) / len(wall_times)
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }
    return metrics


class CustomDistributedSampler(DistributedSampler):
    """
    From torch docs:
    drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas

    We want drop_last = False, but don't want to have extra padding indices. Hence using a custom sampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]")

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil((len(self.dataset) - self.num_replicas) / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas
        else:
            # If we don't drop the last batch, we need to calculate the number of samples per rank.
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


if __name__ == "__main__":
    # Note: This evaluation script saves only model generations. A separate parser is used later to extract predictions and calculate metrics.

    local_rank = setup_ddp()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--few_shot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm", "math", "countdown", "sudoku"],
        default="gsm",
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--checkpoint_path", type=str, default="")
    parser.add_argument("--answer_path", type=str, default=None)
    parser.add_argument("--gen_length", type=int, default=128)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--diffusion_steps", type=int, default=128)
    parser.add_argument("--add_reasoning", action="store_true")
    parser.add_argument("--add_ref", action="store_true")
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--dont_use_box", action="store_true")
    parser.add_argument("--calculate_distance", action="store_true")
    parser.add_argument("--calculate_confidence", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--record_all", action="store_true")
    parser.add_argument("--passk", action="store_true")
    parser.add_argument("--clean_answer", action="store_true")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--remasking", type=str, default="low_confidence")
    parser.add_argument("--pre_answer_keep_mode", type=str, default="prefix")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num_answer_per_question", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    init_seed(args.seed)

    main_print(f"Using diffusion steps: {args.diffusion_steps}, block length: {args.block_length} for gen length: {args.gen_length}")
    num_evals = {"gsm": -1, "math": -1, "countdown": 256, "sudoku": 256}

    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16).to(
        local_rank
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if args.checkpoint_path:
        model = PeftModel.from_pretrained(model, args.checkpoint_path, torch_dtype=torch.bfloat16).to(
            local_rank
        )

        if dist.get_world_size() > 1:
            dist.barrier()  # Make sure all processes are ready
            for param in model.parameters():
                dist.broadcast(param.data, src=0)
            print(f"Rank {local_rank}: Parameters synchronized")

    dataset = DATASET_MAP[args.dataset](
        tokenizer,
        subsample=num_evals[args.dataset],
        num_examples=args.few_shot,
        add_reasoning=True,  # prefill for all models
        split=args.split,
        add_ref=args.add_ref,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=dataset.collate_fn,
    )

    if len(args.checkpoint_path):
        model_name_parts = args.checkpoint_path.split("/")
        model_name_part1 = model_name_parts[-2]
        if args.debug:
            model_name_part1 = "debug/" + model_name_part1
        model_name_part2 = model_name_parts[-1]
        base_path = f"{args.output_dir}/{args.dataset}/{model_name_part1}/{model_name_part2}/{args.remasking}/{args.gen_length}_{args.block_length}_{args.diffusion_steps}"
    else:
        if args.debug:
            model_name = "debug/base_model"
        else:
            model_name = "base_model"
        base_path = f"{args.output_dir}/{args.dataset}/{model_name}/{args.remasking}/{args.gen_length}_{args.block_length}_{args.diffusion_steps}"

    if args.few_shot > 0:
        base_path = base_path + f"_fs{args.few_shot}"

    if len(args.suffix) > 0:
        base_path = base_path + f"_{args.suffix}"

    os.makedirs(base_path, exist_ok=True)
    filename = f"{base_path}/{dist.get_rank()}_generations.json"
    print(f"Saving generations to {filename}")
    if args.calculate_distance:
        distance_dir = f"{base_path}/{dist.get_rank()}_distance"
        os.makedirs(distance_dir, exist_ok=True)
        main_print(f"Saving distances to {distance_dir}")
    else:
        distance_dir = ""
    if args.calculate_confidence:
        confidence_dir = f"{base_path}/{dist.get_rank()}_confidence"
        os.makedirs(confidence_dir, exist_ok=True)
        main_print(f"Saving confidences to {confidence_dir}")
    else:
        confidence_dir = ""

    metrics = evaluate(
        model,
        tokenizer,
        dataloader,
        gen_length=args.gen_length,
        temperature=args.temperature,
        block_length=args.block_length,
        steps=args.diffusion_steps,
        calculate_distance=args.calculate_distance,
        distance_dir=distance_dir,
        calculate_confidence=args.calculate_confidence,
        confidence_dir=confidence_dir,
        num_answer_per_question=args.num_answer_per_question,
        remasking=args.remasking,
        debug=args.debug,
        record_all=args.record_all,
        passk=args.passk,
        answer_path=args.answer_path,
        pre_answer_keep_mode=args.pre_answer_keep_mode,
        clean_answer=args.clean_answer,
        add_ref=args.add_ref,
        dataset=args.dataset,
    )

    if not args.dont_save:
        with open(filename, "w") as f:
            json.dump(
                {
                    "generations": metrics["generations"],
                    "metrics": {
                        "wall_time": metrics["wall_time"],
                        "total_processed": metrics["total_processed"],
                    },
                    "model_path": args.model_path,
                    "checkpoint_path": args.checkpoint_path,
                    "gen_length": args.gen_length,
                    "diffusion_steps": args.diffusion_steps,
                    "block_length": args.block_length,
                },
                f,
                indent=2,
            )
        main_print(f"Generations saved to {filename}")

    cleanup_ddp()