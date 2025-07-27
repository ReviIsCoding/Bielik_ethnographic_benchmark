import pytest
import pandas as pd
from modules.utils import parse_output, build_prompt

def test_parse_output_with_valid_format():
    """ Tests whether parse_output correctly extracts the answer and explanation 
    when both are present in the expected format. """
    
    raw_output = "Answer: C\nExplanation: This is the way."
    answer, explanation = parse_output(raw_output)

    assert answer == "C"
    assert explanation.startswith("This is") 

def test_parse_output_without_answer():
    """Test fallback behavior when no answer (A-D) is explicitly given.
    The function should return fallback strings when 'Answer:' is missing"""

    raw_output = "Explanation: The model could not determine a valid answer."
    answer, explanation = parse_output(raw_output)
    assert answer == "Parsing error"
    assert explanation == "Exception during parsing."

def test_parse_output_unstructured():
    """
    Test edge case where the output is unstructured or inconsistent
    with the prompt format. Parser should return fallback strings.
    """

    raw_output = "C, because that is the way."
    answer, explanation = parse_output(raw_output)
    assert answer == "Parsing error"
    assert explanation == "Exception during parsing."

def test_build_prompt_with_valid_row():
    """ Test that build_prompt correctly formats a prompt from a row of test data."""

    row = pd.Series({
        "Pytanie" : "Jaka gwara używana jest w Poznaniu?",
        "A" : "Śląska",
        "B" : "Kaszubska",
        "C" : "Mazurska",
        "D" : "Poznańska",
        "Pozycja" : "D",
        "Domena" : "Etnologia",
        "Kategoria" : "Gwara"
    })
    prompt = build_prompt(row)
    assert "Jaka gwara używana jest w Poznaniu?" in prompt
    assert "A: Śląska" in prompt
    assert "Answer: [A/B/C/D]" in prompt
    assert "Explanation: [Twoja krótka odpowiedź]" in prompt

def test_build_prompt_with_missing_option():
    """ Test that build prompt handles missing required option.
    The function should raise KeyError if a required key is missing."""

    row = pd.Series({
        "Pytanie": "Który region słynie z oscypków?",
        "A": "Mazury",
        "B": "Kujawy",
        "C": "Podlasie",
        # brak 'D'
        "Pozycja" : "D",
        "Domena" : "Kultura",
        "Kategoria" : "Jedzenie"})

    with pytest.raises(KeyError):
        build_prompt(row)

