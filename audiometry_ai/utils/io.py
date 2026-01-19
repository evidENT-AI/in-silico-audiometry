"""
I/O utilities for audiometry_ai package.

Handles file reading, writing, and format conversions.
"""

from pathlib import Path
from typing import Union, Optional
import pandas as pd


def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str or Path
        Directory path to create

    Returns
    -------
    Path
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe(df: pd.DataFrame,
                   filepath: Union[str, Path],
                   format: str = 'csv',
                   **kwargs) -> None:
    """
    Save a DataFrame to file with format detection.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str or Path
        Output file path
    format : str, default='csv'
        Output format ('csv', 'parquet', 'pickle')
    **kwargs
        Additional arguments passed to the save function
    """
    filepath = Path(filepath)
    ensure_directory_exists(filepath.parent)

    if format == 'csv':
        df.to_csv(filepath, **kwargs)
    elif format == 'parquet':
        df.to_parquet(filepath, **kwargs)
    elif format == 'pickle':
        df.to_pickle(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_dataframe(filepath: Union[str, Path],
                   format: Optional[str] = None,
                   **kwargs) -> pd.DataFrame:
    """
    Load a DataFrame from file with format detection.

    Parameters
    ----------
    filepath : str or Path
        Input file path
    format : str, optional
        File format. If None, inferred from extension
    **kwargs
        Additional arguments passed to the load function

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame
    """
    filepath = Path(filepath)

    if format is None:
        format = filepath.suffix.lstrip('.')

    if format == 'csv':
        return pd.read_csv(filepath, **kwargs)
    elif format in ['parquet', 'pq']:
        return pd.read_parquet(filepath, **kwargs)
    elif format in ['pickle', 'pkl']:
        return pd.read_pickle(filepath, **kwargs)
    else:
        raise ValueError(f"Unsupported format: {format}")
