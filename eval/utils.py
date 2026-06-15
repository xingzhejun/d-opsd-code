import sys
import os
import re
import tiktoken
from parser_helper import remove_boxed, last_boxed_only_string


def main_print(content):
    if int(os.environ["LOCAL_RANK"]) <= 0:
        print(content)

def count_effective_tokens(text):
    if not text:
        return 0
    text = text.replace("<|endoftext|>", "")
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return len(tokens)

def get_parsed_answer(raw_generation, ground_truth):
    parsed_answer = None
    boxed_matches = re.findall(r"\\boxed{(.*?)}", raw_generation)
    effective_tokens = count_effective_tokens(raw_generation)

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
    return parsed_answer, effective_tokens

def get_parsed_answer_math(raw_generation, ground_truth):
    effective_tokens = count_effective_tokens(raw_generation)

    parsed_answer = None
    try:
        parsed_answer = remove_boxed(last_boxed_only_string(raw_generation))
    except:
        parsed_answer = None

    if not parsed_answer:
        answer_match = re.search(r"<answer>(.*?)</answer>", raw_generation, re.DOTALL)
        if answer_match:
            parsed_answer = answer_match.group(1).strip()

    return parsed_answer, effective_tokens

def get_parsed_answer_sudoku(raw_generation, ground_truth, question):
    effective_tokens = count_effective_tokens(raw_generation)
    
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
    
    return solution_str, accuracy, correct_cells, empty_cells, effective_tokens

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

def get_parsed_answer_countdown(raw_generation, ground_truth, question):
    effective_tokens = count_effective_tokens(raw_generation)
    numbers = []
    target = None

    if isinstance(ground_truth, list) and len(ground_truth) == 2:
        numbers = ground_truth[0]
        target = ground_truth[1]
    else:
        # Fallback to parsing from question if ground_truth is not in expected format
        numbers_match = re.search(r"Numbers: \[([\d, ]+)\]", question, re.IGNORECASE)
        if numbers_match:
            numbers_str = numbers_match.group(1)
            numbers = [int(num.strip()) for num in numbers_str.split(",")] # [44, 19, 35] for example

        target_match = re.search(r"Target: (\d+)", question, re.IGNORECASE)
        if target_match:
            target = int(target_match.group(1)) 
        
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
    
    return equation, result, target, effective_tokens, is_valid