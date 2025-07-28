def evaluate_answer(model_answer: str, correct_answer: str) -> str:
    """
    Evaluates the model's answer against the correct answer.
    Returns one of the following:
    - 'prawidłowa' - answer is in the correct format, the same as the correct answer,
    - 'nieprawidłowa' - answer is in the correct format, but not the same as the correct answer,
    - 'brak odpowiedzi' - answer is "Generation error" or "Parsing error",
    - 'odpowiedź niezgodna z oczekiwaniami' - answer is not in the correct format.

    Args:
        model_answer (str): Asnwer given by the evaluated model (e.g. "A", "B", "Generation error", etc.)
        correct_answer (str): correct answer from the test set  (e.g. "A", "B", "C", "D")

    Returns:
        str: Evaluation label
    """
    normalized_model_answer =  model_answer.strip().upper()
    normalized_correct_answer = correct_answer.strip().upper()

    if normalized_model_answer in ["GENERATION ERROR", "PARSING ERROR"]:
        return "brak odpowiedzi"
    elif normalized_model_answer == normalized_correct_answer:
        return 'prawidłowa'
    elif normalized_model_answer in ["A", "B", "C", "D"]:
        return 'nieprawidłowa'
    else:
        return 'odpowiedź niezgodna z oczekiwaniami'
    
def count_evaluation_labels(results:list[dict]) -> dict[str, int]:
    """
    Counts how many times each evaluation label appears in the results.
    
    Each entry in results is a dictionary with the following keys:
    - "question_id": int, identifier of the question
    - 'label': str, evaluation label from the list ['prawidłowa', 'nieprawidłowa', 'brak odpowiedzi', 'odpowiedź niezgodna z oczekiwaniami']

    Args:
        results (list[dict]): List of dictionaries with evaluation results.
    
    Returns:
        dict[str, int]: Dictionary with counts of each evaluation label.
    """
    evaluation_counts = {
        'prawidłowa': 0,
        'nieprawidłowa': 0,
        'brak odpowiedzi': 0,
        'odpowiedź niezgodna z oczekiwaniami': 0
    }

    for entry in results:
        label = entry.get('label')
        if label in evaluation_counts:
            evaluation_counts[label] += 1
        else:
            raise ValueError(f"Unexpected label: {label}")
        
    return evaluation_counts