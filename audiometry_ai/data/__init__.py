"""
Data loading module for NHANES audiometric data.

This module provides tools for downloading NHANES data from the CDC website
and preparing it for prior construction in Bayesian audiometry.

Usage:
------
>>> from audiometry_ai.data import NHANESDownloader
>>>
>>> # Download audiometry and demographics data
>>> downloader = NHANESDownloader(component="Examination")
>>> results = downloader.download_datasets(
...     datasets=["Audiometry", "Audiometry - Tympanometry"],
...     years=["2015-2016", "2017-2018"]
... )
"""

from .nhanes_downloader import NHANESDownloader
from .constants import (
    AUDIOMETRY_DATASETS,
    PRIOR_DATASETS,
    NHANES_AUDIO_CYCLES,
    RECOMMENDED_CYCLES,
    PTA_COLUMNS,
    STANDARD_FREQUENCIES,
    NHANES_FREQUENCIES,
    AGE_GROUPS,
)

__all__ = [
    'NHANESDownloader',
    'AUDIOMETRY_DATASETS',
    'PRIOR_DATASETS',
    'NHANES_AUDIO_CYCLES',
    'RECOMMENDED_CYCLES',
    'PTA_COLUMNS',
    'STANDARD_FREQUENCIES',
    'NHANES_FREQUENCIES',
    'AGE_GROUPS',
]
