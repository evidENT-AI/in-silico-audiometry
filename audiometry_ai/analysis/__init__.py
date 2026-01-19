"""
Analysis module for audiometry results.

This module contains functions for:
- Statistical analysis of test results
- Comparison between different procedures
- Metrics calculation and evaluation
- Phenotype matching (H3 hypothesis)
- Reliability analysis (ICC)
- Bayesian hypothesis testing (Bayes Factors, posteriors, HDI)
"""

from .hearing_level_estimation import *
from .hearing_level_est_mHW import *
from .phenotype_matching import (
    PhenotypeMatching,
    PhenotypeFeatures,
    PhenotypeCentroid,
    cross_validate_matching,
)
from .reliability import compute_icc, bland_altman_stats, ICCResult, BlandAltmanResult
from .bayesian_hypothesis import (
    test_h1_efficiency,
    test_h2_reliability,
    test_h2_reliability_from_differences,
    bayesian_icc,
    run_bayesian_analysis,
    BayesFactorResult,
    PosteriorResult,
    BayesianHypothesisResult,
    format_bayesian_results,
)

__all__ = [
    # Phenotype matching
    'PhenotypeMatching',
    'PhenotypeFeatures',
    'PhenotypeCentroid',
    'cross_validate_matching',
    # Reliability
    'compute_icc',
    'bland_altman_stats',
    'ICCResult',
    'BlandAltmanResult',
    # Bayesian hypothesis testing
    'test_h1_efficiency',
    'test_h2_reliability',
    'test_h2_reliability_from_differences',
    'bayesian_icc',
    'run_bayesian_analysis',
    'BayesFactorResult',
    'PosteriorResult',
    'BayesianHypothesisResult',
    'format_bayesian_results',
]