from typing import Any
from modules.local_backend import run_local_model
from modules.api_backend import run_api_model

def ask_model(prompt: str, config: dict[str, Any]) -> tuple[str, str]:
    """
    Delegates the prompt to the correct backend (local or API) basend on config['api'].

    Args:
        prompt (str): The full prompt to send to the model.
        config (dict): Configuration dictionary with at least:
            - 'api': 'local', 'openAI' or 'google'
            - 'model_id': model name or HF ID
            - additional backend specific options

    Returns:
        tuple[str,  str]: (answer, explanation)
    """

    api_type = config['api']

    if api_type == 'local':
        return run_local_model(prompt, config)
    elif api_type in ["openAI", "google"]:
        return run_api_model(prompt, config)
    else:
        raise NotImplementedError(f"Unsupported API backend: {api_type}") 