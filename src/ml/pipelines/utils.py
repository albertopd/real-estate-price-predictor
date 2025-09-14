from pathlib import Path
import pandas as pd
from utils.logging_utils import setup_logger

logger = setup_logger(__name__)

def combine_datasets(apartment_data_path: Path, house_data_path: Path) -> pd.DataFrame:
    """
    Assemble dataframe by loading the latest parquet file from apartments and houses folders.

    Args:
        raw_dir: Directory containing 'apartments' and 'houses' subdirectories

    Returns:
        Combined dataframe from the latest files in each category
    """
    frames = []

    if not apartment_data_path.exists() and not house_data_path.exists():
        raise RuntimeError("No raw data found to build analysis dataset")

    frames.append(pd.read_parquet(apartment_data_path))
    logger.info(f"Loaded apartments file: {apartment_data_path.name}")

    frames.append(pd.read_parquet(house_data_path))
    logger.info(f"Loaded houses file: {house_data_path.name}")

    combined_df = pd.concat(frames, ignore_index=True)
    logger.info(f"Combined dataset shape: {combined_df.shape}")

    return combined_df

