import pandas as pd
import os

REQUIRED_COLUMNS = ['Pytanie', 'A', 'B', 'C', 'D', 'Pozycja']

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from a file (CSV or XLSX)
    and validates the required columns.

    Args:
        filepath (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Loaded and validated DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the required columns are missing.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file {filepath} does not exist.")
    
    # load the dataset
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, encoding='utf-8')
    elif filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or XLSX file.")
    
    # validate required columns
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"The dataset is missing the following required columns: {missing_columns}")
    
    # drop empty rows
    df.dropna(subset=REQUIRED_COLUMNS, inplace=True)

    return df