import os
import requests
from openai import OpenAI
import google.generativeai as genai
from dotenv import load_dotenv
from typing import Any
from modules.utils import parse_output

# Load environment variables from .env
load_dotenv()

def run_api_model(prompt: str, config: dict[str, Any]) -> tuple[str, str]:
    """
    Executes a prompt using a remote LLM API backend.

    Supported backends (config["api"]):
    - "openAI"
    - "google"
    - "hf_api"

    Args:
        prompt (str): Prompt to send to the model.
        config (dict): Configuration dict including:
            - 'api': backend name
            - 'model_id': model name (e.g. 'gpt-4', 'gemini-pro', HF model ID)
            - 'api_key': optional, else taken from .env
            - 'url': optional custom endpoint
            - 'max_length': optional token limit

    Returns:
        tuple[str, str]: Parsed (answer, explanation)
    """
    api_type = config["api"]
    model_id = config["model_id"]
    max_length = config.get("max_length", 256)

    if api_type == "openAI":

        try:
            api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY"),
            client = OpenAI(api_key = api_key, base_url = config.get("url"))

            response = client.chat.completions.create(
                model = model_id,
                messages = [{"role": "user", "content": prompt}],
                max_tokens = max_length
            )
            raw_output = response.choices[0].message.content.strip()

        except Exception as e:
            print(f"OpenAI error: {e}")
            return "Generation error", "Exception during generation."
        
        return parse_output(raw_output)
    
    elif api_type == "google":
        
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
    
    elif api_type == "hf_api":

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
            print(f"[ERROR] Hugging Face Inference API failed: {e}") 
            return "Generation error", "Exception during generation."
        
        return parse_output(raw_output)
    
    else:
        raise NotImplementedError(f"Unsupported API backend: {api_type}")