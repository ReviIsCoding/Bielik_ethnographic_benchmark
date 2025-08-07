import pytest
from unittest.mock import patch, MagicMock
from modules.local_backend import load_local_model, run_local_model, _local_model_cache

# -------------------------------
# TEST: Loading and cache
# -------------------------------

@patch('modules.local_backend.pipeline')
@patch('modules.local_backend.AutoTokenizer.from_pretrained')
@patch('modules.local_backend.AutoModelForCausalLM.from_pretrained')
def test_load_local_model_basic(mock_model, mock_tokenizer, mock_pipeline):
    """
    Tests the load_local_model function:
    - Checks that the model is loaded and cached correctly,
    - Ensures that the pipeline returns the expected object,
    - Verifies that the cache is updated when the model is loaded.
    """
    _local_model_cache.clear()
    mock_pipe = MagicMock()
    mock_pipeline.return_value = mock_pipe

    result = load_local_model('mock-id', use_q4 = False)
    
    assert result == mock_pipe
    assert _local_model_cache['mock-id'] == mock_pipe

#  Cache is working: no loading again if model was used before
@patch('modules.local_backend.pipeline')
@patch('modules.local_backend.AutoTokenizer.from_pretrained')
@patch('modules.local_backend.AutoModelForCausalLM.from_pretrained')
def test_model_is_cached_on_second_call(mock_model, mock_tokenizer, mock_pipeline):
    """ Test that the model is cached and not loaded again."""
    _local_model_cache.clear()
    mock_pipe = MagicMock()
    mock_pipeline.return_value = mock_pipe

    pipe1 = load_local_model('mock-id', use_q4=False)
    pipe2 = load_local_model('mock-id', use_q4=False)

    assert pipe1 is pipe2
    assert mock_model.call_count == 1  # loaded only once
    assert mock_pipeline.call_count == 1

# -------------------------------
#  TEST: run_local_model is working
# -------------------------------

@patch('modules.local_backend.parse_output', return_value=("A", "explanation"))
@patch('modules.local_backend.load_local_model')
def test_run_local_model_success(mock_load_model, mock_parse):
    """
    Tests the run_local_model function:
    - Verifies that the model pipeline is called with correct arguments,
    - Checks that the output is parsed correctly into answer and explanation,
    - Ensures that the mocked pipeline and parse_output are used as expected.
    """

    mock_pipe = MagicMock(return_value=[{"generated_text": "Answer: A\nExplanation: explanation"}])
    mock_load_model.return_value = mock_pipe

    config = {'model_id': 'mock-id', 'max_length': 100}
    answer, explanation = run_local_model("prompt", config)

    assert answer == "A"
    assert explanation == "explanation"
    mock_pipe.assert_called_once_with("prompt", max_length=100, do_sample=False, truncation=True)

# -------------------------------
#  TEST: Error handling in the pipeline
# -------------------------------

@patch('modules.local_backend.load_local_model')
def test_run_local_model_generation_error(mock_load_model):
    """ Test error handlig when generation error occures."""
    mock_pipe = MagicMock(side_effect=RuntimeError("fail"))
    mock_load_model.return_value = mock_pipe

    config = {'model_id': 'mock-id'}
    answer, explanation = run_local_model("prompt", config)

    assert answer == "Generation error"
    assert "Exception during generation" in explanation

# -------------------------------
# TEST: use_q4 = True uses quant_config
# -------------------------------
@patch('modules.local_backend.pipeline')
@patch('modules.local_backend.AutoTokenizer.from_pretrained')
@patch('modules.local_backend.AutoModelForCausalLM.from_pretrained')
def test_load_local_model_use_q4(mock_model, mock_tokenizer, mock_pipeline):
    """ Test that load_local_model uses quantisation when needed."""
    _local_model_cache.clear()
    mock_pipeline.return_value = MagicMock()

    load_local_model('quant-model', use_q4=True)

    mock_model.assert_called_once()
    args, kwargs = mock_model.call_args
    assert 'quantization_config' in kwargs

# -------------------------------
# TEST: missing model_id → KeyError
# -------------------------------

def test_run_local_model_missing_model_id():
    config = {"max_length": 100}
    with pytest.raises(KeyError):
        run_local_model("prompt", config)