import os
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

    Args:
        prompt (str): Prompt to send to the model.
        config (dict): Configuration dict including:
            - 'api': backend name
            - 'model_id': model name (e.g. 'gpt-4', 'gemini-pro', HF model ID)
            - 'api_key': optional, else taken from .env
            - 'url': optional custom endpoint
            - 'max_new_tokens': optional limit for newly generated tokens (default: 256)

    Returns:
        tuple[str, str]: Parsed (answer, explanation)
    """
    api_type = config["api"]
    model_id = config["model_id"]
    try:
        max_new_tokens = int(config.get("max_new_tokens", 256) or 256)
    except (TypeError, ValueError):
        max_new_tokens = 256

    if api_type == "openAI":

        try:
            api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
            client = OpenAI(api_key = api_key, base_url = config.get("url"))

            response = client.chat.completions.create(
                model = model_id,
                messages = [{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens
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
            response = model.generate_content(
                prompt,
                generation_config={"max_output_tokens": int(max_new_tokens)},
            )
            raw_output = response.text.strip()
        
        except Exception as e:
            print(f"Google Generative AI error: {e}")
            return "Generation error", "Exception during generation."
        
        return parse_output(raw_output)
    
    else:
        raise NotImplementedError(f"Unsupported API backend: {api_type}")