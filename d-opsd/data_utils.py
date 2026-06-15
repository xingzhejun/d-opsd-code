from datasets import load_dataset, Dataset
import pandas as pd
from reward_func import extract_hash_answer

import random
import numpy as np
import torch
import os
import re


## Constant System Prompts
SYSTEM_PROMPT = """
You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}. 
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>
"""

SUDOKU_SYSTEM_PROMPT = """
Please solve the following 4x4 Sudoku puzzle. The puzzle is provided as a 16-character string reading left-to-right, top-to-bottom, where '0' represents empty cells.

Rules:
- Fill empty cells with digits 1-4
- Each row must contain digits 1-4 exactly once
- Each column must contain digits 1-4 exactly once
- Each 2x2 box must contain digits 1-4 exactly once

Important: Your solution must be a COMPLETE 16-character string with only the digits 1-4, representing your final solved grid.

Respond in this exact format:
<reasoning>
Your step-by-step solving process
</reasoning>
<answer>
[16-character solution string with no spaces or separators]
</answer>
"""

CTD_SYSTEM_PROMPT = (
    "Using only the provided numbers, create an arithmetic expression that evaluates to exactly the provided target number. You may use the operations +, -, *, and / as needed, but each number must be used exactly once. Think step-by-step. After reasoning, provide only your final expression inside \\boxed"
    + "{}"
    + " tags without including an equals sign or the target number. For example: \\boxed{a + b * c}"
    + """Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""
)


def extract_reasoning_gsm8k(generated_text):
    """
    Extract the reasoning part from the generated text based on the expected format.
    """
    try:
        # find the first match between ### and #### 
        match = re.search(r"(.*?)(?:####|###)", generated_text, re.DOTALL)
        if match:
            return match.group(1).strip()
            
    except Exception as e:
        print(f"Error extracting reasoning: {e}, Text: {generated_text[:100]}")
    
    return None

def get_gsm8k_questions(split="train", add_ref=False) -> Dataset:
    data = load_dataset("openai/gsm8k", "main")[split]
    if add_ref: # AR-style adding a reference solution to the prompt
        return data.map(
            lambda x: {
                "prompt": [
                    {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["question"]},
                ],
                "teacher_prompt": [
                    {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["question"]
                     + "\n\nHere is a reference solution:\n" + extract_reasoning_gsm8k(x["answer"]) + "\n\nAfter understanding the reference solution, please try to solve this problem using your own approach below:"},
                ],
                "answer": extract_hash_answer(x["answer"]),
            }
        )
    else: 
        return data.map(
            lambda x: {
                "prompt": [
                    {"role": "user", "content": SYSTEM_PROMPT + "\n\n" + x["question"]},
                ],
                "answer": extract_hash_answer(x["answer"]),
            }
        )

def get_countdown_questions(split="train") -> Dataset:
    data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)
    data = data.filter(lambda x: len(x["nums"]) == 3)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{CTD_SYSTEM_PROMPT}\n\nNumbers: {x['nums']}\nTarget: {x['target']}",
                },
            ],
            "target": x["target"],
            "numbers": x["nums"],
        }
    )

def get_sudoku_questions() -> Dataset:
    """Load the Sudoku dataset for training or evaluation."""
    dir_path = os.path.dirname(os.path.abspath(__file__))
    sudoku_file_path = os.path.join(dir_path, "../dataset", "4x4_sudoku_unique_puzzles.csv")
    df = pd.read_csv(sudoku_file_path, dtype={"Puzzle": str, "Solution": str})
    data = Dataset.from_pandas(df)

    return data.map(
        lambda x: {
            "prompt": [
                {
                    "role": "user",
                    "content": f"{SUDOKU_SYSTEM_PROMPT}\n\nSolve the following Sudoku puzzle: {x['Puzzle']}\n",
                },
            ],
            "puzzle": x["Puzzle"],
            "solution": x["Solution"],
        }
    )

def get_math_questions(split="train", add_ref=False) -> Dataset:
    data = load_dataset("ankner/math-500",split=split)  # type: ignore
    if add_ref: # AR-style adding a reference solution to the prompt
        data = data.map(
            lambda x: {  # type: ignore
                "teacher_prompt": [
                    {
                        "role": "user",
                        "content": f"{SYSTEM_PROMPT}\n\n{x['problem']}\n\nHere is a reference solution:\n{x['solution']}\n\nAfter understanding the reference solution, please try to solve this problem using your own approach below:"
                    },
                ],
                "prompt": [
                    {"role": "user", 
                     "content": f"{SYSTEM_PROMPT}\n\n{x['problem']}"
                    },
                ],
                "answer": x["solution"],
            }
        )  # type: ignore
    else:
        data = data.map(
            lambda x: {  # type: ignore
                "prompt": [
                    {
                        "role": "user",
                        "content": f"{SYSTEM_PROMPT}\n\n{x['problem']}",
                    },
                ],
                "answer": x["solution"],
            }
        )  # type: ignore
    return data  # type: ignore