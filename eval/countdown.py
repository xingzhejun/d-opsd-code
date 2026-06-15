import os
import json
from parsers import Parser, evaluate_equation, validate_equation
from gsm8k import GSM8KDataset
from datasets import load_dataset
import warnings

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


class CTDDataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=CTD_SYSTEM_PROMPT,
        subsample=256,
        split="test",
        add_ref=False,
    ):
        if num_examples > 0:
            warnings.warn("num_examples must be 0 for Countdown dataset. Overriding num_examples to 0.")  
        super().__init__(
            tokenizer,
            0,
            add_reasoning,
            system_prompt,
            subsample if split=="test" else 500,
            split,
            add_ref
        )  # num_examples = always 0

    def load_test_dataset(self):
        self.dataset = []
        if self.split == "test":
            dir_path = os.path.dirname(os.path.abspath(__file__))
            countdown_file_path = os.path.join(dir_path, "../dataset", "countdown_cd3_test.jsonl")
            with open(countdown_file_path, "r") as f:
                for line in f:
                    self.dataset.append(json.loads(line))
        else:
            data = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split="train").select(range(1500)) # after the next line filtering, only around 500.
            data = data.filter(lambda x: len(x["nums"]) == 3)
            for item in data:
                self.dataset.append(
                    {
                        "input": ",".join(str(num) for num in item["nums"]),
                        "output": str(item["target"]),
                    }
                )
        print(len(self.dataset), "examples loaded")

    def __getitem__(self, idx):
        target = int(self.dataset[self.subsample[idx].item()]["output"])
        numbers_str = self.dataset[self.subsample[idx].item()]["input"]
        numbers = [int(num) for num in numbers_str.split(",")]
        question = f"Numbers: {numbers}\nTarget: {target}"
        '''
            Numbers: [44, 19, 35]
            Target: 98
        '''
        prompt = self.create_prompt(question, input_reasoning=None)
        return prompt, question, (numbers, target)