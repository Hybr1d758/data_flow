import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import logging
import io
import os
import argparse
import json
from typing import Dict, List, Union


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path: str) -> Dict:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path (str): Path to the configuration file.
    
    Returns:
        Dict: Configuration dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Configuration file not found: {config_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in configuration file: {config_path}")
        raise

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        df = pd.read_csv(filepath)
        logging.info(f"Data loaded successfully from {filepath}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def inspect_data(df: pd.DataFrame) -> None:
    """
    Inspect the data and log basic information.
    
    Args:
        df (pd.DataFrame): Input dataframe.
    """
    logging.info("Data Inspection:")
    logging.info(f"Shape: {df.shape}")
    logging.info("\nFirst few rows:")
    logging.info(df.head().to_string())
    logging.info("\nData info:")
    buffer = io.StringIO()
    df.info(buf=buffer)
    logging.info(buffer.getvalue())
    logging.info("\nSummary statistics:")
    logging.info(df.describe().to_string())

def handle_missing_values(df: pd.DataFrame, strategy: str = 'drop', fill_value: Union[str, float] = None, threshold: float = 0.1) -> pd.DataFrame:
    """
    Identify and handle missing values.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        strategy (str): Strategy to handle missing values ('drop', 'fill', 'forward', 'backward').
        fill_value (Union[str, float]): Value to use for filling if strategy is 'fill'.
        threshold (float): Maximum proportion of missing values allowed for dropping.
    
    Returns:
        pd.DataFrame: Dataframe with handled missing values.
    """
    missing_prop = df.isnull().mean()
    if (missing_prop > threshold).any():
        logging.warning(f"Columns with high missing values (>{threshold*100}%): {missing_prop[missing_prop > threshold].index.tolist()}")
    
    if strategy == 'drop':
        df = df.dropna(thresh=int((1-threshold)*len(df)))
    elif strategy == 'fill':
        df = df.fillna(fill_value)
    elif strategy == 'forward':
        df = df.fillna(method='ffill')
    elif strategy == 'backward':
        df = df.fillna(method='bfill')
    else:
        raise ValueError(f"Invalid strategy: {strategy}")
    
    logging.info(f"Missing values handled using strategy: {strategy}")
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows from the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe.
    
    Returns:
        pd.DataFrame: Dataframe with duplicates removed.
    """
    initial_rows = len(df)
    df = df.drop_duplicates()
    removed_rows = initial_rows - len(df)
    logging.info(f"Removed {removed_rows} duplicate rows")
    return df

def convert_data_types(df: pd.DataFrame, type_dict: Dict[str, str]) -> pd.DataFrame:
    """
    Convert data types of columns based on the provided dictionary.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        type_dict (Dict[str, str]): Dictionary mapping column names to desired data types.
    
    Returns:
        pd.DataFrame: Dataframe with converted data types.
    """
    for col, dtype in type_dict.items():
        try:
            df[col] = df[col].astype(dtype)
        except Exception as e:
            logging.error(f"Error converting {col} to {dtype}: {str(e)}")
    logging.info("Data types converted successfully")
    return df

def remove_outliers(df: pd.DataFrame, columns: List[str], threshold: float = 3) -> pd.DataFrame:
    """
    Detect and remove outliers using Z-score method.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        columns (List[str]): List of columns to check for outliers.
        threshold (float): Z-score threshold for outlier detection.
    
    Returns:
        pd.DataFrame: Dataframe with outliers removed.
    """
    initial_rows = len(df)
    for column in columns:
        if df[column].dtype in [np.float64, np.int64]:
            # Calculate z-scores and convert back to a Series
            z_scores = np.abs(pd.Series(stats.zscore(df[column].dropna()), index=df[column].dropna().index))
            
            # Filter using threshold and handle NaN values
            df = df[(z_scores < threshold) | (z_scores.isna())]

    removed_rows = initial_rows - len(df)
    logging.info(f"Removed {removed_rows} rows containing outliers")
    return df

def validate_and_filter_data(df: pd.DataFrame, validation_dict: Dict[str, Dict]) -> pd.DataFrame:
    """
    Validate data based on provided rules and filter out invalid rows.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        validation_dict (Dict[str, Dict]): Dictionary of validation rules for each column.
    
    Returns:
        pd.DataFrame: Filtered dataframe with only valid data.
    """
    initial_rows = len(df)
    
    for column, rules in validation_dict.items():
        if 'min' in rules:
            df = df[df[column] >= rules['min']]
        if 'max' in rules:
            df = df[df[column] <= rules['max']]
        if 'unique' in rules and rules['unique']:
            df = df.drop_duplicates(subset=[column])
    
    removed_rows = initial_rows - len(df)
    logging.info(f"Data validation removed {removed_rows} invalid rows")
    return df

def save_data(df: pd.DataFrame, filepath: str) -> None:
    """
    Save the cleaned data to a CSV file.
    
    Args:
        df (pd.DataFrame): Dataframe to save.
        filepath (str): Path to save the CSV file.
    """
    try:
        df.to_csv(filepath, index=False)
        logging.info(f"Cleaned data saved to: {filepath}")
    except Exception as e:
        logging.error(f"Error saving data: {str(e)}")
        raise

def clean_data(config: Dict) -> None:
    """
    Main function to run the data cleaning process.
    
    Args:
        config (Dict): Configuration dictionary.
    """
    df = load_data(config['input_filepath'])
    inspect_data(df)
    
    df = handle_missing_values(df, strategy=config['missing_value_strategy'], threshold=config['missing_value_threshold'])
    df = remove_duplicates(df)
    df = convert_data_types(df, config['data_types'])
    df = remove_outliers(df, columns=config['numerical_columns'], threshold=config['outlier_threshold'])
    
    validate_and_filter_data(df, config['validation_rules'])
    
    save_data(df, config['output_filepath'])
    logging.info("Data cleaning process completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pet Adoption Data Cleaning Pipeline")
    parser.add_argument('config', type=str, help='Path to the configuration file')
    args = parser.parse_args()
    
    config = load_config(args.config)
    clean_data(config)
