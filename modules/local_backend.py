import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Any
from modules.utils import parse_output

# Internal cache to avoid reloading models
_local_model_cache: dict[str, Any] = {}

def load_local_model(model_id: str, use_q4: bool = False):
    """
    Loads and returns a text generation pipeline for a local model.
    Models are cached in memory to avoid repeated loading.

    Args:
        model_id (str): Hugging Face model ID.
        use_q4 (bool): Whether to use 4-bit quantization (requires bitsandbytes).

    Returns:
        transformers.Pipeline: Text generation pipeline.
    """
    if model_id in _local_model_cache:
        return _local_model_cache[model_id]
    
    if use_q4:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_4bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            quantization_config=quant_config
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, return_full_text=False)

    _local_model_cache[model_id] = pipe
    return pipe

def run_local_model(prompt: str, config: dict[str, Any]) -> tuple[str, str]:
    """
    Executes a prompt using a local Hugging Face model via pipeline.

    Args:
        prompt (str): The input prompt.
        config (dict): Configuration dict. Expected keys:
            - model_id: Hugging Face model ID
            - max_new_tokens: (optional) new tokens limit
            - use_q4: (optional) whether to use quantization
    
    Returns:
        tuple[str, str]: Parsed (answer, explanation)
    """

    model_id = config["model_id"]
    max_new_tokens = int(config.get("max_new_tokens", 256) or 256)
    use_q4 = config.get("use_q4", False)

    pipe = load_local_model(model_id, use_q4=use_q4)

    try:
        print(f"[Local model] Prompting model with:\n{prompt}")
        response = pipe(
            prompt, 
            max_new_tokens=max_new_tokens, 
            do_sample = False, 
            truncation = True
        )
        raw_output = response[0]["generated_text"].strip()
    except Exception as e:
        print(f"[ERROR] Local model generation failed: {e}")
        return "Generation error", "Exception during generation."
    
    return parse_output(raw_output)