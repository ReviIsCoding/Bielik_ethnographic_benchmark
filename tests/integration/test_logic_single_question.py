import pandas as pd
from unittest.mock import patch
from modules.utils import build_prompt, parse_output
from modules.response_saver import save_raw_results


def test_logic_single_question(tmp_path):
    """ Test logic for a single question:
    build_prompt -> ask_model(mock) -> parse_output -> save_raw_results"""

    # 1. Input - single question DataSeries
    row = pd.Series({
        "Pytanie": "Który region słynie z wypieku sękacza?",
        "A": "Podhale",
        "B":"Mazowsze",
        "C": "Podlasie i Mazury",
        "D": "Kujawy",
        "Pozycja": "C",
        "Domena": "Kulinaria",
        "Kategoria": "Wypieki",
        "Tagi": "ciasta, tradycje, żywność regionalna"
    })

    prompt = build_prompt(row)

    # 2. Mock the ask_model function to simulate LLM response
    fake_response = "Answer: C, Explanation: Sękacz jest tradycyjnym ciastem z Podlasia i Mazur."

    with patch("modules.llm_connector.ask_model", return_value = fake_response):
        #import ask_model from the correct module
        from modules.llm_connector import ask_model

        model_config = {
            "api": "local",
            "model_id": "mock_model",
            "max_length": 256,
            "use_q4": False,
            "api_key": None,
            "url": None
        }

        raw_output = ask_model(prompt, model_config)
        answer, explanation = parse_output(raw_output)

        assert answer == "C"
        assert explanation.startswith("Sękacz jest ")

        # 3. Save results to a temporary path
        output_file = tmp_path / "one_result.json"
        results = [{
            "numer": 0,
            "pytanie": row["Pytanie"],
            "poprawna": row["Pozycja"],
            "odpowiedź": answer,
            "uzasadnienie": explanation,
            "meta": {
                "domena": row["Domena"],
                "kategoria": row["Kategoria"],
                "tagi": row["Tagi"]
            }
        }]

        save_raw_results(results, str(output_file))
        assert output_file.exists()

        # 4. Validate saved structure
        df_loaded = pd.read_json(output_file)
        assert df_loaded.loc[0, "poprawna"] == "C"
        assert df_loaded.loc[0, "odpowiedź"] == "C"
        assert df_loaded.loc[0, "uzasadnienie"].startswith("Sękacz")
