import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Any
from utils import parse_output

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

def ask_model(prompt: str, config: dict[str, Any]) -> tuple[str, str]:
    """
    Generates an answer and explanation using the selected LLM backend.
    
    Args:
        prompt (str): Full formatted prompt including question and multiple-choice options.
        config (dict): Configuration dictionary containing at least:
            - 'api': one of [local, OpenAI, ...]
            - 'model_id' : Hugging Face model ID or API name
            - 'use_q4' : (optional) use quantized model
            - 'max_length' : (optional) max token length

    Returns:
        tuple[str, str]: (anaswer, explanation)
    """
    api_type = config.get('api', 'local')

    if api_type == 'local':
        model_id = config['model_id']
        max_length = config.get('max_length', 512)
        use_q4 = config.get('use_q4', False)

        pipe = load_local_model(model_id, use_q4=use_q4)

        try:
            print(f"[Local model] Prompting model with: \n{prompt}")
            response = pipe(prompt, max_length=max_length, do_sample=False, truncation=True)
            raw_output = response[0]['generated_text'].strip()
        except Exception as e:
            print(f"Error generating response: {e}")
            return "Generation error", "Exception during generation."
        
        
        return parse_output(raw_output)
    
     # TODO: Add support for other APIs (e.g. OpenAI, vLLM, HuggingFace API)
     
    else:
        raise NotImplementedError(f"API backend '{api_type}' is not supported yet.")
