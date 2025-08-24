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

@patch("modules.api_backend.parse_output", return_value=("B", "openai"))
@patch("modules.api_backend.OpenAI")
def test_run_api_model_openai_uses_max_new_tokens(mock_openai, _):
    """Test if run_api_model correctly uses max_new_tokens from config."""
    mc = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock(message=MagicMock(content="Answer: B\nExplanation: openai"))]
    mc.chat.completions.create.return_value = resp
    mock_openai.return_value = mc

    cfg = {"api": "openAI", "model_id": "gpt-4o", "api_key": "x", "max_new_tokens": 77}
    run_api_model("prompt", cfg)

    mc.chat.completions.create.assert_called_once()
    _, kwargs = mc.chat.completions.create.call_args
    
    assert kwargs["max_tokens"] == 77

@patch("modules.api_backend.parse_output", return_value=("C", "google"))
@patch("modules.api_backend.genai.GenerativeModel")
@patch("modules.api_backend.genai.configure")
def test_run_api_model_google_success(_, mock_model_cls, __):
    """Test if run_api_model correctly uses Google GenerativeModel and returns expected output."""
    mock_model = MagicMock()
    mock_model.generate_content.return_value.text = "Answer: C\nExplanation: google"
    mock_model_cls.return_value = mock_model

    cfg = {"api": "google", "model_id": "gemini-1.5-pro", "api_key": "g", "max_new_tokens": 64}
    a, e = run_api_model("p", cfg)

    assert (a, e) == ("C", "google")
    mock_model.generate_content.assert_called_once_with("p", generation_config={"max_output_tokens": 64})


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

