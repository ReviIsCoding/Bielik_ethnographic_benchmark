import pytest
from unittest.mock import patch, MagicMock
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

@patch("modules.llm_connector.load_local_model")
@patch("modules.llm_connector.parse_output")

def test_ask_model_logic_with_mock(mock_parse_output, mock_load_model):
    """
    Test logic of ask_model function localy, without calling the actual model.

    Mocks 'load_local_model' and 'parse_output' to:
    - ensure that the pipeline is loaded correctly,
    - test the integration without external text generation,
    - check if 'ask_model' returns the expected output according to 'parse_output'.

    Also checks if the components are called with the expected parameters.
    """
    # prepare the mock objects
    mock_pipe = MagicMock()
    mock_pipe.return_value = [{'generated_text' : 'A. Explanation about ethnography.'}]
    mock_load_model.return_value = mock_pipe
    mock_parse_output.return_value = ('A', 'Explanation about ethnography.')
    
    config = {
        'api' : 'local',
        'model_id' : 'mock-model',
        'max_length' : 100,
        'use_q4' : False
    }

    prompt = 'What is the ethnographic method? A. Observation B.Interview C. Survey D. Experiment'

    answer, explanation = llm_connector.ask_model(prompt, config)

    # Assert
    mock_load_model.asser_called_once_with('mock-model', use_q4 = False)
    mock_pipe.assert_called_once_with(prompt, max_length=100, do_sample = False, truncation = True)
    mock_parse_output.assert_called_once_with('A. Explanation about ethnography.')
    assert answer == 'A'
    assert explanation == 'Explanation about ethnography.'

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