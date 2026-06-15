import sys
import os
import re
import torch
import torch.nn.functional as F
import numpy as np
import random


'''
Some of them are copied from math500_utils.py
'''

def set_random_seed(seed: int = 42):
    # Set the seed for Python's built-in random module
    random.seed(seed)
    # Set the seed for NumPy
    np.random.seed(seed)
    # Set the seed for PyTorch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior in cuDNN (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main_print(content):
    if int(os.environ["LOCAL_RANK"]) <= 0:
        print(content)

def get_num_transfer_tokens(mask_index, steps):
    """
    Precompute the number of tokens to transition at each step.
    Optimized to be more efficient.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    # Create tensor once and modify in-place
    num_transfer_tokens = base.expand(-1, steps).clone()

    # Handle remainder more efficiently
    if remainder.sum() > 0:
        indices = torch.arange(steps, device=mask_index.device)
        mask = indices.unsqueeze(0) < remainder
        num_transfer_tokens[mask] += 1

    return num_transfer_tokens.to(torch.int64)

def add_gumbel_noise(logits, temperature, dtype):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0.0:
        return logits  # Skip noise when temperature is 0
    logits = logits.to(dtype)
    noise = torch.rand_like(logits, dtype=dtype)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise

def get_all_parsed_answer(generation, answer, dataset):
    if dataset == "gsm8k":
        parsed_answer = get_parsed_answer(generation, answer) 
        try:
            is_correct = parsed_answer is not None and parsed_answer == float(answer)
        except ValueError:
            is_correct = (
                parsed_answer is not None 
                and str(parsed_answer) == answer
            )
    elif dataset == "math":
        parsed_answer = get_parsed_answer_math(generation, answer)
        real_answer = None
        try:
            real_answer = remove_boxed(last_boxed_only_string(answer))
        except:
            real_answer = None

        if not real_answer:
            answer_match = re.search(r"<answer>(.*?)</answer>", answer, re.DOTALL)
            if answer_match:
                real_answer = answer_match.group(1).strip()
        is_correct = False
        if parsed_answer is not None:
            is_correct = is_equiv(parsed_answer, real_answer)
        # main_print(f'real_answer is: {real_answer}')    
    return parsed_answer, is_correct

def get_parsed_answer(raw_generation, ground_truth):
    parsed_answer = None
    boxed_matches = re.findall(r"\\boxed{(.*?)}", raw_generation)

    if boxed_matches:
        for boxed_content in boxed_matches:
            # boxed_content = boxed_content.strip()
            boxed_content = boxed_content.strip().replace(",", "") 
            if boxed_content and boxed_content != "..." and not re.match(r"^\.+$", boxed_content):
                try:
                    parsed_answer = float(boxed_content)
                    break
                except ValueError:
                    numbers = re.findall(r"-?\d+\.?\d*", boxed_content)
                    if numbers:
                        try:
                            parsed_answer = float(numbers[0])
                            break
                        except ValueError:
                            pass
    if parsed_answer is None:
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
        if answer_match:
            answer_text = answer_match.group(1).strip()
            if answer_text:
                try:
                    parsed_answer = float(answer_text)
                except ValueError:
                    numbers = re.findall(r"-?\d+\.?\d*", answer_text)
                    if numbers:
                        try:
                            parsed_answer = float(numbers[-1])
                        except ValueError:
                            pass
    return parsed_answer

def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string

def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string

def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except AssertionError:
        return string

def is_equiv(str1, str2, verbose=False):
    if type(str1) == float or type(str2) == float:
        try:
            return abs(float(str1) - float(str2)) < 1e-6
        except:
            return False
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2

def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return string

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    try:
        assert s[: len(left)] == left
        assert s[-1] == "}"

        return s[len(left) : -1]
    except:
        return s

def get_parsed_answer_math(raw_generation, ground_truth):
    parsed_answer = None
    try:
        parsed_answer = remove_boxed(last_boxed_only_string(raw_generation))
    except:
        parsed_answer = None

    if not parsed_answer:
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
        if answer_match:
            parsed_answer = answer_match.group(1).strip()

    return parsed_answer

def get_parsed_answer_sudoku(raw_generation, ground_truth, question):
    puzzle_str = ""
    if len(question) >= 16 and all(c.isdigit() or c == "0" for c in question[:16]):
        puzzle_str = question[:16]
    else:
        match = re.search(r"Sudoku puzzle: ([0-9]{16})", question)
        if match:
            puzzle_str = match.group(1)
    assert len(puzzle_str) == 16, f"Invalid puzzle string: {puzzle_str}"
    empty_indices = [i for i in range(16) if puzzle_str[i] == "0"]
    empty_cells = len(empty_indices)

    # Extract solution using regex patterns
    solution_str = ""
    patterns = [
        r"<answer>.*?```\s*([\d\s]+)```",
        r"<answer>(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|</answer>)",
        r"</answer>\s*(.*?)(?:<\|eot_id\|>|<\|endoftext\|>|$)",
        r".*?(\d{16})\s*</answer>",
        r"\b(\d{16})\b",
    ]

    for pattern in patterns:
        if solution_str:
            break
        match = re.search(pattern, raw_generation, re.DOTALL)
        if match and match.group(1).strip():
            solution_str = match.group(1).strip()
    solution_str = re.sub(r"\s", "", solution_str)

    # Handle solution length
    if not solution_str:
        correct_cells = 0
    else:
        if len(solution_str) < 16:
            solution_str = solution_str + "0" * (16 - len(solution_str))
        elif len(solution_str) > 16:
            solution_str = solution_str[:16]
        correct_cells = sum(1 for i in empty_indices if solution_str[i] == ground_truth[i])

    accuracy = correct_cells / empty_cells if empty_cells > 0 else 0.0
    
    return solution_str, accuracy

def validate_equation(equation_str, available_numbers):
    """Validate that equation only uses available numbers and each number once."""
    try:
        numbers_in_eq = [int(n) for n in re.findall(r"\d+", equation_str)]
        available_numbers = sorted(available_numbers)
        numbers_in_eq = sorted(numbers_in_eq)
        return numbers_in_eq == available_numbers
    except:
        return False

def evaluate_equation(equation_str):
    """Safely evaluate the arithmetic equation."""
    try:
        allowed_pattern = r"^[\d+\-*/().\s]+$"
        if not re.match(allowed_pattern, equation_str):
            raise ValueError("Invalid characters in equation.")
        result = eval(equation_str.strip(), {"__builtins__": None}, {})
        return result
    except Exception:
        return float("Inf")

def get_parsed_answer_countdown(raw_generation, numbers, target):
    equation = ""
    try:
        equation = remove_boxed(last_boxed_only_string(raw_generation))
    except:
        # Try to extract from answer tags
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
        if answer_match:
            equation = answer_match.group(1).strip()
        else:
            equation = raw_generation
    # Replace LaTeX operators with Python operators
    equation = equation.replace(r"\div", "/").replace(r"\times", "*").replace(r"\cdot", "*")
    # Check for equation with equals sign and extract only the expression part
    equation_match = re.search(r"([0-9+\-*/() ]+)=[0-9. ]+", equation)
    if equation_match:
        equation = equation_match.group(1).strip()
    
    is_correct = False
    result = None
    # Validate and evaluate the equation
    is_valid = validate_equation(equation, numbers)
    if is_valid:
        result = evaluate_equation(equation)
        if target is not None and abs(result - target) < 1e-5:
            is_correct = True
    return equation, is_correct

@torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=256,
    block_length=32,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
    mask_id=126336,
    eos_token_id=126081,
    debug1=False,
    fp16=False # self.args.fp16
):
    """
    Optimized version of the generate function.
    """
    with torch.cuda.amp.autocast(enabled=True):
        batch_size = prompt.shape[0]
        dtype = model.dtype
        x = torch.full(
            (batch_size, prompt.shape[1] + gen_length), mask_id, dtype=torch.long, device=model.device
            )
        x[:, : prompt.shape[1]] = prompt.clone()

        prompt_index = torch.zeros_like(x, dtype=torch.bool)
        prompt_index[:, : prompt.shape[1]] = True
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        steps_per_block = max(1, steps // num_blocks)
        x_trajectory = []
        
        for num_block in range(num_blocks):
            start_idx = prompt.shape[1] + num_block * block_length
            end_idx = prompt.shape[1] + (num_block + 1) * block_length
            block_mask_index = x[:, start_idx:end_idx] == mask_id
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

            for i in range(steps_per_block):
                mask_index = x == mask_id

                current_seq = x[0]
                eos_positions = (current_seq == eos_token_id).nonzero(as_tuple=True)[0]
                if eos_positions.numel() == 0:
                    should_append = True
                else:
                    first_eos_pos = eos_positions[0].item()
                    should_append = (current_seq[:first_eos_pos] == mask_id).any().item()
                if num_block == num_blocks - 1:
                    should_append = False

                if should_append:
                    x_trajectory.append(x.clone())
                    # if debug1:
                    #     main_print(f'block {num_block}, step {i}: Appended trajectory with shape {x.shape}')
                        # main_print(f'x is now: {x}')

                if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
                    with torch.cuda.amp.autocast(enabled=fp16):
                        if cfg_scale > 0.0:
                            un_x = x.clone()
                            un_x[prompt_index] = mask_id
                            x_ = torch.cat([x, un_x], dim=0)
                            # Get logits in a single forward pass
                            logits = model(x_).logits
                            logits, un_logits = torch.chunk(logits, 2, dim=0)
                            logits = un_logits + (cfg_scale + 1) * (logits - un_logits)
                        else:
                            logits = model(x).logits

                        # Apply Gumbel noise for sampling
                        logits_with_noise = add_gumbel_noise(logits, temperature=temperature, dtype=dtype)
                        x0 = torch.argmax(logits_with_noise, dim=-1)

                        # Handle remasking strategy
                        if remasking == "low_confidence":
                            p = F.softmax(logits.to(dtype), dim=-1)
                            x0_p = torch.squeeze(torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1)
                        elif remasking == "random":
                            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                        else:
                            raise NotImplementedError(remasking)

                        # Ensure we don't process tokens beyond the current block
                        x0_p[:, end_idx:] = -np.inf

                        # Update masked tokens
                        x0 = torch.where(mask_index, x0, x)
                        confidence = torch.where(mask_index, x0_p, -np.inf)

                        # Select tokens to transfer based on confidence
                        transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                        for j in range(confidence.shape[0]):
                            num_tokens = num_transfer_tokens[j, i].item()
                            if num_tokens > 0:
                                _, select_index = torch.topk(confidence[j], k=num_tokens)
                                transfer_index[j, select_index] = True
                                
                        x[transfer_index] = x0[transfer_index]
        
        return x, x_trajectory