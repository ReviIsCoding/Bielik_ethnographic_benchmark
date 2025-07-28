import pytest
from modules.scorer import evaluate_answer, count_evaluation_labels

# test for evaluate_answer()

def test_evaluate_answer_correct():
    "Test case where model answer matches correct answer"""
    assert evaluate_answer("A", "A") == 'prawidłowa'
    assert evaluate_answer("B", 'b ') == 'prawidłowa'
    assert evaluate_answer("c", "C ") == 'prawidłowa'

def test_evaluate_answer_incorrect():
    """ Test case where model asnwer is in correct format, 
    but does not match the correct answer."""
    assert evaluate_answer("A", "B") == 'nieprawidłowa'
    assert evaluate_answer("B", ' c') == 'nieprawidłowa'
    assert evaluate_answer("c ", "A") == 'nieprawidłowa'

def test_evaluate_answer_missing():
    " Test case where model answer is a generation or parsing error."
    assert evaluate_answer("Generation error", "A") == 'brak odpowiedzi'
    assert evaluate_answer("Parsing error", "B ") == 'brak odpowiedzi'
    assert evaluate_answer("  parsing error  ", "D") == "brak odpowiedzi"

def test_evaluate_answer_invalid_format():
    "Test case where model answer is not in the expected format."
    assert evaluate_answer("E", "A") == 'odpowiedź niezgodna z oczekiwaniami'
    assert evaluate_answer("1", "B") == 'odpowiedź niezgodna z oczekiwaniami'
    assert evaluate_answer("Tak", "C") == 'odpowiedź niezgodna z oczekiwaniami'
    assert evaluate_answer("", "D") == 'odpowiedź niezgodna z oczekiwaniami'

# test for count_evaluation_labels()

def test_count_evaluation_labels_basic():
    "Test case with a simpe list of results."
    results = [
        {"question_id": 1, "label": "prawidłowa"},
        {"question_id": 2, "label": "nieprawidłowa"},
        {"question_id": 3, "label": "brak odpowiedzi"},
        {"question_id": 4, "label": "odpowiedź niezgodna z oczekiwaniami"},
        {"question_id": 5, "label": "prawidłowa"},
        {"question_id": 6, "label": "nieprawidłowa"},
        {"question_id": 7, "label": "prawidłowa"},
        {"question_id": 8, "label": "prawidłowa"},
        {"question_id": 9, "label": "nieprawidłowa"},
        {"question_id": 10, "label": "prawidłowa"}
    ]

    count = count_evaluation_labels(results)
    assert count == {
        'prawidłowa': 5,
        'nieprawidłowa': 3,
        'brak odpowiedzi': 1,
        'odpowiedź niezgodna z oczekiwaniami': 1
    }

def test_count_evaluation_labels_invalid_label():
    "Test if function raises ValueError for unexpected labels."
    results = [
        {"question_id": 1, "label": "prawidłowa"},
        {"question_id": 2, "label": "nieprawidłowa"},
        {"question_id": 3, "label": "brak odpowiedzi"},
        {"question_id": 4, "label": "odpowiedź niezgodna z oczekiwaniami"},
        {"question_id": 5, "label": "serdelek"}  # Invalid label
    ]
    with pytest.raises(ValueError, match= "Unexpected label: serdelek"):
        count_evaluation_labels(results)
