import json
import re
import os
import glob
from collections import defaultdict
from parser_helper import is_equiv
from utils import get_parsed_answer, get_parsed_answer_math, get_parsed_answer_sudoku, get_parsed_answer_countdown


def parse_gsm_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data
    # data['generations'] = data['generations'][7:11]
    
    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        ground_truth = item.get("ground_truth")
        raw_generation = item.get("generations", "")
        question = item.get("question", "")

        parsed_answer, effective_tokens = get_parsed_answer(raw_generation, ground_truth)
        
        total_effective_tokens += effective_tokens

        is_correct = parsed_answer is not None and parsed_answer == ground_truth
        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
    )

def parse_math_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")

        # Count effective tokens
        parsed_answer, effective_tokens = get_parsed_answer_math(raw_generation, ground_truth)
        total_effective_tokens += effective_tokens

        is_correct = False
        if parsed_answer is not None:
            is_correct = is_equiv(parsed_answer, ground_truth)

        if is_correct:
            total_correct += 1

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": parsed_answer,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
    )

def parse_countdown_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct = 0
    total_processed = 0
    total_effective_tokens = 0

    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", [])
        generated_text = item.get("generations", "")
        
        equation, result, target, effective_tokens, is_valid = get_parsed_answer_countdown(generated_text, ground_truth, question)
        total_effective_tokens += effective_tokens
        is_correct = False
        if is_valid:
            if target is not None and abs(result - target) < 1e-5:
                is_correct = True
                total_correct += 1

        processed_items.append(
            {
                "question": question,
                "extracted_answer": equation,
                "evaluation_result": result,
                "ground_truth": ground_truth,
                "is_correct": is_correct,
                "effective_tokens": effective_tokens,
            }
        )

    return (
        total_correct,
        total_processed,
        processed_items,
        total_effective_tokens,
    )

def parse_sudoku_answers(json_path=None, json_data=None):
    if json_path:
        with open(json_path, "r") as file:
            data = json.load(file)
    else:
        data = json_data

    total_correct_cells = total_empty_cells = total_processed = 0
    total_effective_tokens = 0
    processed_items = []

    for item in data.get("generations", []):
        total_processed += 1
        question = item.get("question", "")
        ground_truth = item.get("ground_truth", "")
        raw_generation = item.get("generations", "")

        # Count effective tokens
        solution_str, accuracy, correct_cells, empty_cells, effective_tokens = get_parsed_answer_sudoku(raw_generation, ground_truth, question)
        total_effective_tokens += effective_tokens

        total_correct_cells += correct_cells
        total_empty_cells += empty_cells

        processed_items.append(
            {
                "question": question,
                "raw_generation": raw_generation,
                "extracted_answer": solution_str,
                "ground_truth": ground_truth,
                "empty_cells": empty_cells,
                "correct_cells": correct_cells,
                "accuracy": accuracy,
                "effective_tokens": effective_tokens,
            }
        )
    return (
        total_correct_cells,
        total_empty_cells,
        processed_items,
        total_effective_tokens * 8,
    )

def extract_setup_name(filename):
    """Extract the setup name from the filename."""
    match = re.match(r"(.+)_\d+_generations\.json$", filename)
    if match:
        return match.group(1)
    return None

def aggregate_results(directory=".", setup_name=None):
    """Aggregate results from all JSON files and save detailed results."""
    # Find all JSON files matching the pattern
    json_files = glob.glob(os.path.join(directory, "*_generations.json"))
    print('json_files:', json_files)

    # Dictionary to store aggregated results by setup
    setups = defaultdict(
        lambda: {
            "correct": 0,
            "processed": 0,
            "accuracy": 0.0,
            "questions": [],
            "total_effective_tokens": 0,
        }
    )

    # You can ignore it. It's just for pre-caculating the statistics of sudoku accuracy distribution.
    def compute_percentiles(values, percentiles):
        """Compute percentile values with linear interpolation."""
        if not values:
            return {p: 0.0 for p in percentiles}

        sorted_values = sorted(values)
        n = len(sorted_values)
        results = {}

        for p in percentiles:
            if n == 1:
                results[p] = sorted_values[0]
                continue

            rank = (n - 1) * (p / 100.0)
            lower = int(rank)
            upper = min(lower + 1, n - 1)
            weight = rank - lower
            results[p] = sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

        return results

    for json_file in json_files:
        filename = os.path.basename(json_file)

        if setup_name:
            # print(f"Processing {filename}...")
            if "gsm" in setup_name:
                (
                    correct,
                    processed,
                    detailed_results,
                    total_effective_tokens,
                ) = parse_gsm_answers(json_path=json_file)
            elif "math" in setup_name:
                (
                    correct,
                    processed,
                    detailed_results,
                    total_effective_tokens,
                ) = parse_math_answers(json_path=json_file)
            elif "countdown" in setup_name:
                (
                    correct,
                    processed,
                    detailed_results,
                    total_effective_tokens,
                ) = parse_countdown_answers(json_path=json_file)
            elif "sudoku" in setup_name:
                (
                    correct,
                    processed,
                    detailed_results,
                    total_effective_tokens,
                ) = parse_sudoku_answers(json_path=json_file)

            setups[setup_name]["correct"] += correct
            setups[setup_name]["processed"] += processed
            setups[setup_name]["total_effective_tokens"] += total_effective_tokens
            setups[setup_name]["questions"].extend(detailed_results)

    # Calculate final accuracy and save results
    for setup, results in sorted(setups.items()):
        results["accuracy"] = (
            results["correct"] / results["processed"] * 100 if results["processed"] > 0 else 0
        )
        results["avg_effective_tokens"] = (
            results["total_effective_tokens"] / results["processed"] if len(results["questions"]) > 0 else 0
        )
    # Header
    header_format = "{:<40} {:>12} {:>25}"
    print(header_format.format("Setup (task_model_seqlen_diffusteps)", "Accuracy", "Avg Effective Tokens"))
    print("-" * 80)

    # Data rows
    row_format = "{:<40} {:>11.2f}% {:>25.2f}"
    for setup, results in sorted(setups.items()):
        print(row_format.format(setup, results["accuracy"], results["avg_effective_tokens"]))

    print("=" * 180)

    # if setup_name == "sudoku":
    #     percentile_labels = list(range(10, 100, 10))
    #     sudoku_row_format = "{:<40} " + " ".join(["{:>9.2f}%"] * len(percentile_labels))
    #     print("Sudoku accuracy percentiles (per-sample):")
    #     print("{:<40} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9} {:>9}".format(
    #         "Setup",
    #         "P10",
    #         "P20",
    #         "P30",
    #         "P40",
    #         "P50",
    #         "P60",
    #         "P70",
    #         "P80",
    #         "P90",
    #     ))
    #     print("-" * 140)
    #     for setup, results in sorted(setups.items()):
    #         sudoku_accuracies = [item["accuracy"] * 100 for item in results["questions"] if item.get("accuracy") is not None]
    #         p = compute_percentiles(sudoku_accuracies, percentile_labels)
    #         print(sudoku_row_format.format(
    #             setup,
    #             p[10],
    #             p[20],
    #             p[30],
    #             p[40],
    #             p[50],
    #             p[60],
    #             p[70],
    #             p[80],
    #             p[90],
    #         ))


if __name__ == "__main__":
    directory_list = ["path/to/your/generations"] # TODO: Update this with the actual path to your generations
    for directory in directory_list:
        aggregate_results(directory=directory,
                        setup_name="gsm")
    # for i in range(25, 501, 25):
    #     aggregate_results(directory=f"path/to/your/i-generations", # TODO: Update this with the actual path to your i-generations
    #                         setup_name="gsm")
