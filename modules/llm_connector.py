import os
import torch
import requests
from openai import OpenAI
import google.generativeai as genai
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import Any
from modules.utils import parse_output
from dotenv import load_dotenv

load_dotenv()

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
            - 'api': one of ['local', 'openAI', 'google', 'hf_api']
            - 'model_id' : Hugging Face model ID or API name
            - other API-specific keys (e.g., 'api_key', 'url', 'use_q4', etc.),
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
    
    elif api_type == 'openAI':
        try:
            client = OpenAI(
                api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY"),
                base_url=config.get("url") # optional: for Azure or custom
            )
            response = client.chat.completions.create(
                model = config["model_id"],
                messages = [{"role": "user", "content": prompt}],
                max_tokens = config.get("max_length", 256)
            )
            raw_output = response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI error: {e}")
            return "Generation error", "Exception during generation."
        
        return parse_output(raw_output)
    
    elif api_type == 'google':
        try:
            api_key = config.get("api_key") or os.getenv("GOOGLE_API_KEY")
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name=config["model_id"])
            response = model.generate_content(prompt)
            raw_output = response.text.strip()

        except Exception as e:
            print(f"Google Generative AI error: {e}")
            return "Generation error", "Exception during generation."
        
        return parse_output(raw_output)

    elif api_type == 'hf_api':
        try:
            api_key = config.get("api_key") or os.getenv("HF_API_KEY")
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            endpoint = config.get("url", f"https://api-inference.huggingface.co/models/{config['model_id']}")
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": config.get("max_length", 512),
                    "do_sample": False,
                    "return_full_text": False
                }
            }
            response = requests.post(endpoint, headers=headers, json=payload)
            response.raise_for_status()
            raw_output = response.json()[0]["generated_text"].strip()
        except Exception as e:
            print(f"Hugging Face API error: {e}") 
            return "Generation error", "Exception during generation."
        
        return parse_output(raw_output)

    else:
        raise NotImplementedError(f"API backend '{api_type}' is not supported yet.")
