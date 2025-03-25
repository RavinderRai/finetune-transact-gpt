import os
import pandas as pd
import kagglehub
import logging
from typing import Optional

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ingest_transactions_dataset(
    dataset_slug: str = "ismetsemedov/transactions",
    file_name: str = "synthetic_fraud_data.csv",
    sample_size: Optional[int] = 100,
    chunk_size: int = 100_000
) -> pd.DataFrame:
    """
    Downloads and loads a sample of the transactions dataset from KaggleHub.

    Args:
        dataset_slug (str): The KaggleHub dataset identifier.
        file_name (str): The expected CSV file name inside the dataset directory.
        sample_size (Optional[int]): Number of rows to load. If None, loads full file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    logger.info("Downloading dataset from KaggleHub...")
    dataset_dir = kagglehub.dataset_download(dataset_slug)
    
    logger.debug(f"Files in dataset: {os.listdir(dataset_dir)}")
    
    file_path = os.path.join(dataset_dir, file_name)

    logger.info(f"Loading dataset from: {file_path}")
    if sample_size < 100000:
        df = pd.read_csv(file_path, nrows=sample_size)
    else:
        logger.info(f"Loading full or large portion of the dataset in chunks of {chunk_size} rows...")
        chunk_iter = pd.read_csv(file_path, chunksize=chunk_size)
        df = pd.concat(chunk_iter, ignore_index=True)

    logger.info(f"Loaded DataFrame with shape: {df.shape}")
    return df

def random_sample(df: pd.DataFrame, n: int = 10000, output_path: str = "data/sampled_data.csv") -> pd.DataFrame:
    """
    Randomly samples rows from a DataFrame and saves to CSV.

    Args:
        df (pd.DataFrame): Input full DataFrame.
        n (int): Number of rows to sample.
        output_path (str): Path to save the sampled CSV.

    Returns:
        pd.DataFrame: The sampled DataFrame.
    """
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_sample = df.sample(n=n, random_state=42)
    df_sample.to_csv(output_path, index=False)
    return df_sample


if __name__ == "__main__":
    # Note the total dataset size is almost 3GB with shape (7483766, 24)
    df = ingest_transactions_dataset(sample_size=100000) # 100000 is probably sufficient
    df_sampled = random_sample(df, n=200)
    print("The first few rows of the DataFrame are:")
    print(df_sampled.head())
