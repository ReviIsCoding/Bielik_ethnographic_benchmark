import pytest
from unittest.mock import patch, MagicMock
from modules import llm_connector


@patch('modules.llm_connector.pipeline')
@patch('modules.llm_connector.AutoTokenizer.from_pretrained')
@patch('modules.llm_connector.AutoModelForCausalLM.from_pretrained')

def test_load_local_model_basic_pipeline(mock_model, mock_tokenizer, mock_pipeline):
    """
    Test if load_local_model creates a pipeline and caches output without downloading real model.
    """
    mock_pipe_instance = MagicMock()
    mock_pipeline.return_value = mock_pipe_instance
    llm_connector._local_model_cache.clear() # Clear cache before the test

    result = llm_connector.load_local_model('mock-model', use_q4 = False)
    
    mock_model.assert_called_once_with(
        'mock-model',
        device_map = 'auto',
        trust_remote_code = True,
        torch_dtype = llm_connector.torch.float16
    )
    mock_tokenizer.assert_called_once_with('mock-model')
    mock_pipeline.assert_called_once()
    assert result == mock_pipe_instance
    assert llm_connector._local_model_cache['mock-model'] == mock_pipe_instance

@patch('modules.llm_connector.pipeline')
@patch('modules.llm_connector.AutoTokenizer.from_pretrained')
@patch('modules.llm_connector.AutoModelForCausalLM.from_pretrained')

def test_load_local_model_caching(mock_model, mock_tokenizer, mock_pipeline):
    """
    Test if load_local_model uses caching when called again.
    """
    mock_pipe_instance = MagicMock()
    mock_pipeline.return_value = mock_pipe_instance
    llm_connector._local_model_cache.clear() # Clear cache before the test
    
    pipe1 = llm_connector.load_local_model('mock-model', use_q4 = False)
    pipe2 = llm_connector.load_local_model('mock-model', use_q4 = False)

    assert pipe1 is pipe2 # Ensures the same instance is returned
    assert mock_model.call_count == 1
    assert mock_pipeline.call_count == 1

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
    mock_load_model.assert_called_once_with('mock-model', use_q4 = False)
    mock_pipe.assert_called_once_with(prompt, max_length=100, do_sample = False, truncation = True)
    mock_parse_output.assert_called_once_with('A. Explanation about ethnography.')
    assert answer == 'A'
    assert explanation == 'Explanation about ethnography.'


@patch("modules.llm_connector.parse_output", return_value = ("B", "Explanation from OpenAI."))
@patch("modules.llm_connector.OpenAI")
def test_ask_model_openai(mock_openai_class, mock_parse_output):
    """ Tests ask_model() with mock_openai_class, 
        returns mocked answer.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Answer: B\nExplanation: Explanation from OpenAI."))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    config = {"api": "openAI", "model_id": "openai-model", "api_key": "sk-test"}
    answer, explanation = llm_connector.ask_model("test prompt", config)

    assert answer == "B"
    assert explanation == "Explanation from OpenAI."

@patch("modules.llm_connector.parse_output", return_value=("C", "Explanation from Google."))
@patch("modules.llm_connector.genai.GenerativeModel")
@patch("modules.llm_connector.genai.configure")
def test_ask_model_google(mock_configure, mock_generative_model_class, mock_parse_output):
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = "Answer: C\nExplanation: Explanation from Google."
    mock_generative_model_class.return_value = mock_model

    config = {"api": "google", "model_id": "gemini_model", "api_key": "gapi-test"}
    answer, explanation = llm_connector.ask_model("test prompt", config)

    assert answer == "C"
    assert explanation == "Explanation from Google."

@patch("modules.llm_connector.parse_output", return_value=("D", "Explanation from HF API."))
@patch("modules.llm_connector.requests.post")
def test_ask_model_hf_api(mock_post, mock_parse_output):
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = [{"generated_text": "Answer: D\nExplanation: Explanation from HF API."}]
    mock_post.return_value = mock_response

    config = {"api": "hf_api", "model_id": "hf/test-model", "api_key": "hf-token"}
    answer, explanation = llm_connector.ask_model("test prompt", config)

    assert answer == "D"
    assert explanation == "Explanation from HF API."

def test_ask_model_unsupported_api():
    """
    Test if unsupported API raises NotImplementedError.
    """

    config = {
        'api': 'vllm',
        'model_id' : "model-x"
    }
    with pytest.raises(NotImplementedError):
        llm_connector.ask_model("What is the ethnographic method?", config)