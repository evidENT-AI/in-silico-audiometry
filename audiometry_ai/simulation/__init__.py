"""
Simulation module for audiometry testing.

This module contains functions and classes for:
- Simulating listener responses
- Generating hearing profiles and audiograms
- Modeling psychometric functions
- Phenotype-based population generation (9 GMM-derived phenotypes)
"""

from .hearing_level_gen import generate_clipped_data
from .response_model import HearingResponseModel
from .phenotypes import (
    PhenotypeGenerator,
    PsychometricParameterGenerator,
    PhenotypeDefinition,
    PHENOTYPE_DEFINITIONS,
    FREQUENCIES,
    USE_APPROXIMATE_GMM,
    get_phenotype_names,
    get_phenotype_categories,
    get_phenotype_proportions,
    validate_gmm_models,
    get_gmm_model,
    GMMModel,
)

__all__ = [
    "generate_clipped_data",
    "HearingResponseModel",
    "PhenotypeGenerator",
    "PsychometricParameterGenerator",
    "PhenotypeDefinition",
    "PHENOTYPE_DEFINITIONS",
    "FREQUENCIES",
    "USE_APPROXIMATE_GMM",
    "get_phenotype_names",
    "get_phenotype_categories",
    "get_phenotype_proportions",
    "validate_gmm_models",
    "get_gmm_model",
    "GMMModel",
]