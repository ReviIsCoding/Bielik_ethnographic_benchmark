import argparse
import time
from pathlib import Path
from modules.dataset_loader import load_dataset
from modules.llm_connector import load_local_model, ask_model
from modules.utils import parse_output, build_prompt
from modules.response_saver import save_raw_results

def main():
    """ Main function that loads the dataset, configures the model, 
    generates answers for each question using the model,
    and saves the raw results for further evaluation.
    """

    parser = argparse.ArgumentParser(description = "Ethnographic Benchmark Runner")
    parser.add_argument("--test", type=str, required=True, help="Path to the test dataset file (.csv/.xlsx)")
    parser.add_argument("--results", type=str, required=True, help="Path tp save raw results (.json)")
    parser.add_argument("--llm", type=str, required=True, help="Model identifier (local od API)")
    parser.add_argument("--llm_name", type=str, required=True, help="Friendly model name for reports")
    parser.add_argument("--api", type=str, required=True, help="API type: local | openAI | vllm")
    parser.add_argument("--url", type=str, default=None, help="API URL (if applicable)")
    parser.add_argument("--key", type=str, default=None, help="API key (if applicable)")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum length of the model's response")
    parser.add_argument("--use_q4", action='store_true', help="Use quantized model (local only)")
    parser.add_argument("--interval", type=int, default=1, help= "Delay between questions in seconds")

    args = parser.parse_args()

    test_data = load_dataset(args.test)
    results = []
    start_time = time.time()

    # Model config passed to ask_model()
    model_config = {
        "api" : args.api,
        "model_id" : args.llm,
        "max_length" : args.max_length,
        "use_q4" : args.use_q4,
        "api_key" : args.key,
        "url" : args.url
    }

    for idx, row in test_data.iterrows():
        prompt = build_prompt(row)
        try:
            # TODO: Extend ask_model in llm_connector.py tp handle API-based models
            raw_output = ask_model(prompt, model_config)
            answer,explanation = parse_output(raw_output)
        except Exception as e:
            print(f"Error processing question {idx}: {e}")
            answer, explanation = "Generation error", "Exception during processing"

        results.append({
            "numer" : idx,
            "pytanie": row["Pytanie"],
            "poprawna": row["Pozycja"],
            "odpowiedź": answer,
            "uzasadnienie": explanation,
            "meta": {
                "domena": row.get("Domena", ""),
                "kategoria": row.get("Kategoria", ""),
                "tagi": row.get("Tagi", "")
            }
        })

        if args.interval > 0:
            time.sleep(args.interval)

        save_raw_results(results, args.results)
        total_time = time.time() - start_time

        print (f"Finished {len(results)} questions in {total_time:.2f} seconds. Results saved to: {args.results}")

        # TODO : Add summary of execution stats here if needed (e.g. total questions, time)
              
if __name__ == "__main__":
    main()

