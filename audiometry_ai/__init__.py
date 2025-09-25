"""
Audiometry AI - Bayesian Optimization of Pure-Tone Audiometry Through In-Silico Modelling
"""

__version__ = "0.1.0"

# Import main classes and functions for easy access
from .simulation.hearing_level_gen import generate_clipped_data
from .procedures.bsa_mhw import ModifiedHughsonWestlakeAudiometry
from .procedures.basic_bayes import BayesianPureToneAudiometry

__all__ = [
    "generate_clipped_data",
    "ModifiedHughsonWestlakeAudiometry",
    "BayesianPureToneAudiometry"
]