import json
from pathlib import Path
from modules.response_saver import save_raw_results 

def test_save_raw_results_creates_valid_json(tmp_path):
    """ Tests that save_raw_results creates a valid
    JSON file with correct content."""

    # Given sample data
    results = [
        {"question_number": 1,
         "question": "Kim był Oskar Kolberg?",
         "correct_answer": "C",
         "model_answer": "C",
         "model_explanation": "Oskar Kolberg był polskim etnografem i folklorystą, który zbierał i opisywał polskie tradycje ludowe.",
         "meta":{
            "domena": "Etnografia",
            "kategoria": "Postacie historyczne",
            "tagi": ["XIX wiek", "etnografia", "folklorystyka"]}
        }]
    output_path = tmp_path /"test_output.json"
    
    # When: savingto file
    save_raw_results(results, str(output_path))

    #Then: file should exist and contain valid JSON

    assert output_path.exists(), "Output file was not created."

    with open(output_path, encoding='utf-8') as f:
        saved_data = json.load(f)

    assert isinstance(saved_data, list), "Saved data is not a list."
    assert saved_data[0]["question_number"] == 1, "Question number does not match."
    assert saved_data[0]["model_answer"] == "C", "Model answer does not match."
    assert "model_explanation" in saved_data[0], "Model explanation is missing."

    