import pytest
from modules.llm_connector import ask_model

def test_ask_model_unsupported_api():
    """
    Tests that ask_model raises NotImplementedError for unsupported API types.
    """
    config = {'api': 'vllm', 'model_id': 'test'}
    with pytest.raises(NotImplementedError):
        ask_model('dummy', config)

def test_ask_model_delegates_to_local(monkeypatch):
    """
    Tests that ask_model delegates to run_local_model when 'api' is set to 'local'.
    Verifies that the returned answer and explanation match the mocked local backend.
    """
    def mock_run_local(prompt, config):
        return "A", "local explanation"
    
    monkeypatch.setattr("modules.llm_connector.run_local_model", mock_run_local)
    config = {'api': 'local', 'model_id': 'mock-model'}
    answer, explanation = ask_model("prompt", config)

    assert answer == 'A'
    assert explanation == 'local explanation'

def test_ask_model_delegates_to_api(monkeypatch):
    """
    Tests that ask_model delegates to run_api_model when 'api' is set to a supported API type.
    Verifies that the returned answer and explanation match the mocked API backend.
    """
    def mock_run_api(prompt, config):
        return "B", "api explanation"
    
    monkeypatch.setattr("modules.llm_connector.run_api_model", mock_run_api)
    config = {'api': 'openAI', 'model_id': 'gpt-4'}
    answer, explanation = ask_model("prompt", config)

    assert answer == "B"
    assert explanation == 'api explanation'

def test_ask_model_missing_model_id_raises():
    """
    Test if ask_model raises KeyError when 'model_id' is missing in config.
    """
    config = {'api': 'local'}  # missing model_id
    with pytest.raises(KeyError):
        ask_model("prompt", config)

def test_ask_model_google_api_delegation(monkeypatch):
    """
    Test if ask_model delegates correctly to run_api_model when using Google API.
    """
    monkeypatch.setattr("modules.llm_connector.run_api_model", lambda p, c: ("C", "response from google"))
    config = {"api": "google", "model_id": "gemini"}
    answer, explanation = ask_model("prompt", config)

    assert answer == "C"
    assert explanation == "response from google"

def test_ask_model_backend_error_propagation(monkeypatch):
    """
    Test if ask_model propagates exceptions thrown by the backend function.
    """
    def raise_error(*args, **kwargs):
        raise RuntimeError("backend failed")

    monkeypatch.setattr("modules.llm_connector.run_local_model", raise_error)
    config = {"api": "local", "model_id": "broken-model"}

    with pytest.raises(RuntimeError, match="backend failed"):
        ask_model("prompt", config)

def test_ask_model_missing_api_key():
    """
    Test if ask_model raises KeyError when 'api' key is missing in config.
    """
    config = {"model_id": "test-model"}
    with pytest.raises(KeyError):
        ask_model("prompt", config)

def test_ask_model_non_string_api_type():
    """
    Test if ask_model raises NotImplementedError when api type is not a string.
    """
    config = {"api": 123, "model_id": "test"}
    with pytest.raises(NotImplementedError):
        ask_model("prompt", config)