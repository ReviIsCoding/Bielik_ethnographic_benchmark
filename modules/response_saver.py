import json
import os
from typing import Any


def save_raw_results(results:list[dict[str, Any]], output_path: str) -> None:
    """
    Save raw model answers to a JSON file.

    Each entry in "results" should include:
    - question_number (int)
    - question (str)
    - correct_answer (str)
    - model_answer (str)
    - model_explanation (str)

    Args:
        results (list): List of dictionaries containing model answers.
        output_path (str): Path to the output JSON file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to {output_path}")