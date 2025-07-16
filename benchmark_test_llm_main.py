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
    
if __name__ == "__main__":
    main()
        
