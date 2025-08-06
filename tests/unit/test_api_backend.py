import pytest
from unittest.mock import patch, MagicMock
from modules.api_backend import run_api_model

@patch("modules.api_backend.parse_output", return_value = ("B", "openai explanation"))
@patch("modules.api_backend.OpenAI")
def test_run_api_model_openai(mock_openai_class, mock_parse):
    """
    Tests the run_api_model function for the OpenAI backend:
    - Mocks the OpenAI client and its response,
    - Verifies that the function correctly parses the model's output,
    - Checks that the returned answer and explanation match the expected values.
    """
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices =  [MagicMock(message = MagicMock(content = "Answer: B\nExplanation: openai explanation" ))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_class.return_value = mock_client

    config = {"api": "openAI", "model_id": "gpt-4", "api_key": "x"}
    answer, explanation = run_api_model("prompt", config)

    assert answer == "B"
    assert explanation == "openai explanation"

@patch("modules.api_backend.parse_output", return_value = ("C", "google explanation"))
@patch("modules.api_backend.genai.GenerativeModel")
@patch("modules.api_backend.genai.configure")
def test_run_api_model_google(mock_configure, mock_model_class, mock_parse):
    """
    Tests the run_api_model function for the Google backend:
    - Mocks the Google GenerativeModel and its response,
    - Verifies that the function correctly parses the model's output,
    - Checks that the returned answer and explanation match the expected values.
    """
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = "Answer: C\nExplanation: google explanation"
    mock_model_class.return_value = mock_model

    config = {'api': 'google', 'model_id': 'menini-pro', 'api_key': 'y'}
    answer, explanation = run_api_model("prompt", config)

    assert answer == "C"
    assert explanation == 'google explanation'

@patch("modules.api_backend.parse_output", return_value = ("D", "hf explanation"))
@patch("modules.api_backend.requests.post")
def test_run_api_model_hf_api(mock_post, mock_parse):
    """
    Tests the run_api_model function for the Hugging Face API backend:
    - Mocks the requests.post call and its response,
    - Verifies that the function correctly parses the model's output,
    - Checks that the returned answer and explanation match the expected values.
    """
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = [{"generated_text": "Answer: D\nExplanation: hf explanation"}]
    mock_post.return_value = mock_response

    config = {"api": "hf_api", "model_id": "mock-model", "api_key": "z"}
    answer, explanation = run_api_model("prompt", config)

    assert answer == "D"
    assert explanation == "hf explanation"


@patch("modules.api_backend.requests.post", side_effect=ConnectionError("Network unreachable"))
def test_run_api_model_hf_api_connection_error(mock_post):
    """
    Test if run_api_model handles network connection errors for HuggingFace API.
    """
    config = {"api": "hf_api", "model_id": "test"}
    answer, explanation = run_api_model("prompt", config)
    assert answer == "Generation error"
    assert "Exception during generation" in explanation

@patch("modules.api_backend.OpenAI")
def test_run_api_model_openai_error(mock_openai):
    """
    Test if run_api_model handles exceptions thrown by OpenAI client.
    """
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("Unauthorized")
    mock_openai.return_value = mock_client

    config = {"api": "openAI", "model_id": "gpt-4", "api_key": "bad_key"}
    answer, explanation = run_api_model("prompt", config)
    assert answer == "Generation error"
    assert "Exception during generation" in explanation


@patch("modules.api_backend.genai.GenerativeModel")
@patch("modules.api_backend.genai.configure")
def test_run_api_model_google_error(mock_configure, mock_model):
    """
    Test if run_api_model handles exceptions from Google Generative AI.
    """
    mock_model.return_value.generate_content.side_effect = Exception("Bad credentials")
    config = {"api": "google", "model_id": "gemini", "api_key": "wrong"}
    answer, explanation = run_api_model("prompt", config)
    assert answer == "Generation error"
    assert "Exception during generation" in explanation

def test_run_api_model_missing_model_id():
    """
    Test if run_api_model raises KeyError when model_id is missing from config.
    """
    config = {"api": "openAI"}
    with pytest.raises(KeyError):
        run_api_model("prompt", config)

@patch("modules.api_backend.requests.post")
def test_run_api_model_hf_api_unexpected_response(mock_post):
    """
    Test if run_api_model handles unexpected HF API response structure (no 'generated_text').
    """
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = [{}]  # missing "generated_text"
    mock_post.return_value = mock_response

    config = {"api": "hf_api", "model_id": "test", "api_key": "x"}
    answer, explanation = run_api_model("prompt", config)
    assert answer == "Generation error"
    assert "Exception during generation" in explanation