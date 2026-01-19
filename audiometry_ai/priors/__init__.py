"""
Prior construction module for Bayesian audiometry.

This module provides:
- NHANES-derived empirical priors for hearing thresholds
- Hierarchical prior conditioning based on covariates
- Literature-derived priors for psychometric parameters

Usage:
------
>>> from audiometry_ai.priors import NHANESPriorBuilder, get_threshold_prior
>>>
>>> # Build priors from NHANES data
>>> builder = NHANESPriorBuilder(data_dir="data/nhanes")
>>> builder.build_all_priors()
>>>
>>> # Get prior for a specific individual
>>> prior = get_threshold_prior(age=65, sex="male", diabetes=True)
"""

from .nhanes_priors import NHANESPriorBuilder, get_threshold_prior
from .conditioning import (
    PriorConditioner,
    apply_age_conditioning,
    apply_sex_conditioning,
    apply_diabetes_conditioning,
    apply_cardiovascular_conditioning,
    apply_noise_exposure_conditioning,
    apply_tympanometry_conditioning,
)

__all__ = [
    'NHANESPriorBuilder',
    'get_threshold_prior',
    'PriorConditioner',
    'apply_age_conditioning',
    'apply_sex_conditioning',
    'apply_diabetes_conditioning',
    'apply_cardiovascular_conditioning',
    'apply_noise_exposure_conditioning',
    'apply_tympanometry_conditioning',
]
