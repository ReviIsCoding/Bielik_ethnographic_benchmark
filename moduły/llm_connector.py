import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Any

_local_model_cache = {}

def load_local_model(model_id: str, use_q4: bool = False) -> Any:
    """
    Loads and returns a text generation pipeline for a local model.
    
    Args: 
        model_id (str): The identifier for the model to load.
        use_q4 (bool): Whether to use quantization for the model.
        
    Returns:
        transformers.pipeline: A text generation pipeline.
    """
    if model_id in _local_model_cache:
        return _local_model_cache[model_id]
    
    if use_q4:
        from transformers import BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(load_in_4bit = True)
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
            torch_dtype=torch.float16
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        return_full_text=False
        )
    _local_model_cache[model_id] = pipe
    return pipe

def ask_model(question: str, config: dict[str, Any]) -> tuple[str, str]:
    
