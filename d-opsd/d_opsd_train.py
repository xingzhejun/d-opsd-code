import torch
import wandb
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from trl import TrlParser, ModelConfig
from peft import LoraConfig
import warnings

# Custom imports
from d_opsd_trainer import dOPSDTrainer
from d_opsd_config import dOPSDConfig
from reward_func import (
    xmlcount_reward_func,
    soft_format_reward_func,
    strict_format_reward_func,
    int_reward_func,
    correctness_reward_func,
    countdown_reward_func,
    correctness_reward_func_math,
    sudoku_reward_func,
    boxed_and_answer_tags_format_reward,
)
from data_utils import (
    get_gsm8k_questions,
    get_countdown_questions,
    get_sudoku_questions,
    get_math_questions,
)
from utils import set_random_seed


def main(opsd_config, model_config):
    # Set seed for reproducibility
    set_random_seed(opsd_config.seed)

    # Load dataset based on configuration
    if opsd_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions(split="train", add_ref=opsd_config.add_ref)
        reward_functions = [
            xmlcount_reward_func,
            soft_format_reward_func,
            strict_format_reward_func,
            int_reward_func,
            correctness_reward_func,
        ]
    elif opsd_config.dataset == "countdown":
        dataset = get_countdown_questions("train")
        reward_functions = [countdown_reward_func]
    elif opsd_config.dataset == "sudoku":
        dataset = get_sudoku_questions()
        reward_functions = [sudoku_reward_func]
    elif opsd_config.dataset == "math":
        dataset = get_math_questions("train", add_ref=opsd_config.add_ref)
        reward_functions = [
            correctness_reward_func_math,
            boxed_and_answer_tags_format_reward,
        ]
    # Shuffle dataset with fixed seed for reproducibility
    dataset = dataset.shuffle(seed=opsd_config.seed)

    # Split dataset if needed
    if opsd_config.dataset in ["countdown", "sudoku"]:
        train_set = dataset.select(range(0, len(dataset) - 500))  # Leave last 500 for evaluation
    else:
        train_set = dataset

    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        opsd_config.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    ).to(device)

    tokenizer = AutoTokenizer.from_pretrained(opsd_config.model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # Configure LoRA for parameter-efficient fine-tuning
    peft_config = LoraConfig(
        r=model_config.lora_r,
        lora_alpha=model_config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
        task_type="CAUSAL_LM",
        lora_dropout=model_config.lora_dropout,
    )
    # Initialize and run trainer
    trainer = dOPSDTrainer(
        args=opsd_config,
        model=model,
        peft_config=peft_config,
        reward_funcs=reward_functions,
        train_dataset=train_set,
    )

    if opsd_config.save_steps % opsd_config.num_iterations != 0:
        warnings.warn(
            f"save_steps ({opsd_config.save_steps}) is not divisible by num_iterations ({opsd_config.num_iterations}). If resuming training from a checkpoint, you might need to manually specify the checkpoint where the training step is divisible by {grpo_config.num_iterations}."
        )

    trainer.train()


if __name__ == "__main__":
    parser = TrlParser((dOPSDConfig, ModelConfig))
    opsd_config, model_config = parser.parse_args_and_config()
    main(opsd_config=opsd_config, model_config=model_config)