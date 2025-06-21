import pytest
from modules import llm_connector

def test_load_local_model_basic_pipeline():
    """
    Test the basic functionality of loading a local model pipeline.
    """
    pipe = llm_connector.load_local_model("sshleifer/tiny-gpt2")
    assert pipe is not None
    assert callable(pipe)

def test_load_local_model_caching():
    """
    Test that the local model is cached and reused.
    """
    pipe1 = llm_connector.load_local_model("sshleifer/tiny-gpt2")
    pipe2 = llm_connector.load_local_model("sshleifer/tiny-gpt2")
    assert pipe1 is pipe2 # Ensures the same instance is returned

def test_ask_model_with_valid_prompt(monkeypatch):
    """ Test ask_model in local mode,
    mocking parse_output to return a fixed answer and explanation. 
    """
    def dumy_parser(text):
        return "A", "This is a dummy explanation."
    monkeypatch.setattr(llm_connector, "parse_output", dumy_parser)

    config = {
        'api': 'local',
        'model_id': 'sshleifer/tiny-gpt2',
        'max_length': 20,
        'use_q4': False
        }

    prompt = "What is the ethnographic method? A) Observation B) Interview C) Survey D) Experiment"
    answer, explanation = llm_connector.ask_model(prompt, config)
    assert answer == "A"
    assert explanation == "This is a dummy explanation."

def test_ask_model_unsupported_api():
    """
    Test if using API raises an error.
    """

    config = {
        'api': 'openAI',
        'model_id' : "some model"
    }
    with pytest.raises(NotImplementedError):
        llm_connector.ask_model("What is the ethnographic method?", config)