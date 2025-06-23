import pytest
import pandas as pd
import openpyxl
from modules import dataset_loader
import os

REQUIRED_COLUMNS = dataset_loader.REQUIRED_COLUMNS

def create_temp_csv(tmp_path, data: pd.DataFrame, filename = 'test.csv'):
    """Create a temporary CSV file for testing."""

    path = tmp_path / filename
    data.to_csv(path, index=False)
    return path

def create_temp_xlsx(tmp_path, data: pd.DataFrame, filename = 'test.xlsx'):
    """Create a temporary Excel file for testing."""

    path = tmp_path / filename
    data.to_excel(path, index=False)
    return path

def test_load_valid_csv_dataset(tmp_path):
    """ Tests if a valid CSV dataset can be loaded correctly."""
    
    df_input = pd.DataFrame([
        {col: f"sample {col}" for col in REQUIRED_COLUMNS}
    ])
    csv_file = create_temp_csv(tmp_path, df_input)

    df = dataset_loader.load_dataset(str(csv_file))

    assert isinstance(df, pd.DataFrame) # Check if the result is a DataFrame
    assert list(df.columns) == REQUIRED_COLUMNS # Check if the columns match
    assert len(df) == 1 # Check if the DataFrame has one row

def test_load_valid_xlsx_dataset(tmp_path):
    """ Tests if a valid Excel dataset can be loaded correctly."""
    
    df_input = pd.DataFrame([
       {col: f"example {col}" for col in REQUIRED_COLUMNS}
   ])
    xlsx_file = create_temp_xlsx(tmp_path, df_input)

    df = dataset_loader.load_dataset(str(xlsx_file))

    assert isinstance(df, pd.DataFrame) # Check if the result is a DataFrame
    assert list(df.columns) == REQUIRED_COLUMNS # Check if the columns match
    assert len(df) == 1 # Check if the DataFrame has one row

def test_missing_required_columns_raises(tmp_path):
    """ Tests if loading a dataset with missing columns raises an error."""

    df_input = pd.DataFrame({
        "Pytanie" : ["Q1"],
        "A" : ['a'],
        "B" : ['b'] # Missing required columns C, D
    })

    csv_file = create_temp_csv(tmp_path, df_input)

    with pytest.raises(ValueError, match=r"The dataset is missing the following required columns"):
        dataset_loader.load_dataset(str(csv_file))

def test_file_not_found():
    """ Tests if loading a non-existent file raises a FileNotFoundError."""

    with pytest.raises(FileNotFoundError):
        dataset_loader.load_dataset("non_existent_file.csv")

def test_unsupported_format(tmp_path):
    """ Tests if loading a file with an unsupported format raises a ValueError."""

    file = tmp_path / "data.json"
    file.write_text('{"sample" : 123}') 

    with pytest.raises(ValueError, match = 'Unsupported file format'):
        dataset_loader.load_dataset(str(file))