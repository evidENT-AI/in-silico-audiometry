"""
Audiometric phenotype definitions for simulation.

This module defines hearing phenotypes based on GMM clusters from the
RNENT clinical dataset (54,927 patients). The 9-cluster model was
derived using Gaussian Mixture Modeling on air-conduction thresholds.

The system supports two modes:
1. APPROXIMATE MODE (current): Uses approximate GMM with estimated covariances
2. FULL GMM MODE (future): Uses exact GMM model from collaborator

To switch to full GMM mode:
- Replace data/phenotype_gmms/gmm_9cluster_rnent.pkl with collaborator's model
- Set USE_APPROXIMATE_GMM = False
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to GMM model files
GMM_MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "phenotype_gmms"
GMM_MODEL_PATH = GMM_MODEL_DIR / "gmm_9cluster_rnent.pkl"
GMM_SKLEARN_PATH = GMM_MODEL_DIR / "gmm_9cluster_sklearn.pkl"

# Flag to indicate we're using approximate model (will be updated when real model arrives)
USE_APPROXIMATE_GMM = True

# Standard audiometric frequencies (matching RNENT GMM)
FREQUENCIES = [250, 500, 1000, 2000, 4000, 8000]

# Extended frequencies (for compatibility)
FREQUENCIES_EXTENDED = [250, 500, 1000, 2000, 3000, 4000, 6000, 8000]


# =============================================================================
# PHENOTYPE DEFINITIONS (9 GMM-derived clusters)
# =============================================================================

@dataclass
class PhenotypeDefinition:
    """Definition of a hearing phenotype from GMM clustering."""
    name: str
    cluster_id: int  # GMM cluster index (0-8)
    category: str  # Clinical category
    n_target: int  # Target sample size for Stage 1 (total = 2200)
    description: str
    mean_thresholds: Dict[int, float] = field(default_factory=dict)
    proportion: float = 0.0  # Proportion in clinical population


# The 9 phenotypes from RNENT GMM analysis
# Proportions and n_target scaled to sum to 2200
PHENOTYPE_DEFINITIONS = {
    # Cluster 1: Moderate sloping (presbycusis) - 11%
    'moderate_sloping': PhenotypeDefinition(
        name='moderate_sloping',
        cluster_id=0,
        category='presbycusis',
        n_target=242,  # 11% of 2200
        description='Moderate sloping loss typical of presbycusis (40-78 dB)',
        mean_thresholds={250: 41, 500: 45, 1000: 50, 2000: 55, 4000: 64, 8000: 78},
        proportion=0.11
    ),

    # Cluster 2: Mild sloping - 8%
    'mild_sloping': PhenotypeDefinition(
        name='mild_sloping',
        cluster_id=1,
        category='presbycusis',
        n_target=176,  # 8% of 2200
        description='Mild sloping sensorineural loss (34-56 dB)',
        mean_thresholds={250: 34, 500: 37, 1000: 41, 2000: 41, 4000: 48, 8000: 56},
        proportion=0.08
    ),

    # Cluster 3: Severe/profound - 3%
    'severe_profound': PhenotypeDefinition(
        name='severe_profound',
        cluster_id=2,
        category='severe',
        n_target=66,  # 3% of 2200
        description='Severe to profound SNHL (81-120 dB)',
        mean_thresholds={250: 81, 500: 93, 1000: 106, 2000: 116, 4000: 120, 8000: 108},
        proportion=0.03
    ),

    # Cluster 4: Near-normal with mild HF - 12%
    'near_normal_mild_hf': PhenotypeDefinition(
        name='near_normal_mild_hf',
        cluster_id=3,
        category='mild',
        n_target=264,  # 12% of 2200
        description='Near-normal thresholds with mild high-frequency loss (16-38 dB)',
        mean_thresholds={250: 16, 500: 16, 1000: 15, 2000: 16, 4000: 25, 8000: 38},
        proportion=0.12
    ),

    # Cluster 5: Moderate high-frequency - 8%
    'moderate_high_freq': PhenotypeDefinition(
        name='moderate_high_freq',
        cluster_id=4,
        category='noise_induced',
        n_target=176,  # 8% of 2200
        description='Moderate high-frequency accentuated loss (24-74 dB)',
        mean_thresholds={250: 24, 500: 27, 1000: 33, 2000: 47, 4000: 62, 8000: 74},
        proportion=0.08
    ),

    # Cluster 6: Moderate-severe - 10%
    'moderate_severe': PhenotypeDefinition(
        name='moderate_severe',
        cluster_id=5,
        category='presbycusis',
        n_target=220,  # 10% of 2200
        description='Moderate-severe flat/sloping loss (68-94 dB)',
        mean_thresholds={250: 68, 500: 70, 1000: 72, 2000: 74, 4000: 83, 8000: 94},
        proportion=0.10
    ),

    # Cluster 7: Mild with HF drop - 20%
    'mild_hf_drop': PhenotypeDefinition(
        name='mild_hf_drop',
        cluster_id=6,
        category='presbycusis',
        n_target=440,  # 20% of 2200
        description='Mild loss with high-frequency drop (16-58 dB)',
        mean_thresholds={250: 16, 500: 17, 1000: 20, 2000: 28, 4000: 48, 8000: 58},
        proportion=0.20
    ),

    # Cluster 8: Steeply sloping (ski-slope) - 15%
    'ski_slope': PhenotypeDefinition(
        name='ski_slope',
        cluster_id=7,
        category='presbycusis',
        n_target=330,  # 15% of 2200
        description='Steeply sloping ski-slope configuration (42-101 dB)',
        mean_thresholds={250: 42, 500: 48, 1000: 60, 2000: 76, 4000: 96, 8000: 101},
        proportion=0.15
    ),

    # Cluster 9: Normal hearing - 13%
    'normal_hearing': PhenotypeDefinition(
        name='normal_hearing',
        cluster_id=8,
        category='normal',
        n_target=286,  # 13% of 2200
        description='Normal hearing thresholds (8-12 dB)',
        mean_thresholds={250: 10, 500: 9, 1000: 9, 2000: 8, 4000: 10, 8000: 12},
        proportion=0.13
    ),
}

# Verify total = 2200
_total = sum(p.n_target for p in PHENOTYPE_DEFINITIONS.values())
assert _total == 2200, f"Total phenotype count should be 2200, got {_total}"


# =============================================================================
# GMM MODEL LOADING
# =============================================================================

class GMMModel:
    """
    Wrapper for GMM model that handles both approximate and full models.
    """

    def __init__(self, model_path: Path = GMM_MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.sklearn_model = None
        self.is_loaded = False
        self.is_approximate = True

    def load(self):
        """Load GMM model from disk."""
        if self.is_loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"GMM model not found at {self.model_path}\n"
                "Run 'python scripts/create_approximate_gmm.py' to create it."
            )

        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)

        self.is_approximate = self.model.get('is_approximate', False)
        self.is_loaded = True

        # Try to load sklearn model for sampling
        if GMM_SKLEARN_PATH.exists():
            with open(GMM_SKLEARN_PATH, 'rb') as f:
                self.sklearn_model = pickle.load(f)

    def sample(self, n_samples: int = 1, random_state: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample audiograms from the GMM.

        Returns
        -------
        tuple
            (audiograms, cluster_labels) where audiograms is shape (n_samples, 6)
        """
        if not self.is_loaded:
            self.load()

        if self.sklearn_model is not None:
            # Use sklearn for sampling
            rng = np.random.default_rng(random_state)
            # sklearn's sample doesn't take our random state directly
            audiograms, labels = self.sklearn_model.sample(n_samples)
        else:
            # Manual sampling from model parameters
            rng = np.random.default_rng(random_state)
            weights = self.model['weights']
            means = self.model['means']
            covariances = self.model['covariances']

            labels = rng.choice(len(weights), size=n_samples, p=weights)
            audiograms = np.zeros((n_samples, means.shape[1]))

            for i, k in enumerate(labels):
                audiograms[i] = rng.multivariate_normal(means[k], covariances[k])

        # Clip to valid range and round to 5 dB steps
        audiograms = np.clip(audiograms, -10, 120)
        audiograms = np.round(audiograms / 5) * 5

        return audiograms, labels

    def sample_from_cluster(
        self,
        cluster_id: int,
        n_samples: int = 1,
        random_state: int = None
    ) -> np.ndarray:
        """Sample audiograms from a specific cluster."""
        if not self.is_loaded:
            self.load()

        rng = np.random.default_rng(random_state)
        means = self.model['means']
        covariances = self.model['covariances']

        audiograms = rng.multivariate_normal(
            means[cluster_id],
            covariances[cluster_id],
            size=n_samples
        )

        # Clip and round
        audiograms = np.clip(audiograms, -10, 120)
        audiograms = np.round(audiograms / 5) * 5

        return audiograms

    @property
    def n_components(self) -> int:
        if not self.is_loaded:
            self.load()
        return self.model['n_components']

    @property
    def means(self) -> np.ndarray:
        if not self.is_loaded:
            self.load()
        return self.model['means']

    @property
    def weights(self) -> np.ndarray:
        if not self.is_loaded:
            self.load()
        return self.model['weights']


# Global GMM model instance
_gmm_model = None


def get_gmm_model() -> GMMModel:
    """Get the global GMM model instance."""
    global _gmm_model
    if _gmm_model is None:
        _gmm_model = GMMModel()
    return _gmm_model


# =============================================================================
# PHENOTYPE GENERATOR
# =============================================================================

class PhenotypeGenerator:
    """
    Generate audiograms for each phenotype using the GMM model.
    """

    def __init__(self):
        self.gmm = get_gmm_model()
        self._cluster_to_phenotype = {
            p.cluster_id: name for name, p in PHENOTYPE_DEFINITIONS.items()
        }
        self._phenotype_to_cluster = {
            name: p.cluster_id for name, p in PHENOTYPE_DEFINITIONS.items()
        }

    def generate_audiogram(
        self,
        phenotype_name: str,
        frequencies: List[int] = FREQUENCIES,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[int, float]:
        """
        Generate a single audiogram for the given phenotype.

        Parameters
        ----------
        phenotype_name : str
            Name of the phenotype (must be in PHENOTYPE_DEFINITIONS)
        frequencies : list of int
            Frequencies to generate thresholds for
        rng : np.random.Generator, optional
            Random number generator for reproducibility

        Returns
        -------
        dict
            {frequency: threshold_dB_HL}
        """
        if rng is None:
            rng = np.random.default_rng()

        if phenotype_name not in PHENOTYPE_DEFINITIONS:
            raise ValueError(f"Unknown phenotype: {phenotype_name}")

        cluster_id = self._phenotype_to_cluster[phenotype_name]
        seed = rng.integers(0, 2**31)

        # Sample from GMM cluster
        audiogram_array = self.gmm.sample_from_cluster(
            cluster_id, n_samples=1, random_state=seed
        )[0]

        # Convert to dictionary
        audiogram = {
            freq: float(audiogram_array[i])
            for i, freq in enumerate(FREQUENCIES)
        }

        # Handle extended frequencies if requested (interpolate)
        if set(frequencies) != set(FREQUENCIES):
            audiogram = self._interpolate_frequencies(audiogram, frequencies)

        return audiogram

    def _interpolate_frequencies(
        self,
        audiogram: Dict[int, float],
        target_frequencies: List[int]
    ) -> Dict[int, float]:
        """Interpolate audiogram to target frequencies."""
        result = {}
        source_freqs = sorted(audiogram.keys())
        source_thresholds = [audiogram[f] for f in source_freqs]

        for freq in target_frequencies:
            if freq in audiogram:
                result[freq] = audiogram[freq]
            else:
                # Linear interpolation in log-frequency space
                log_freq = np.log2(freq)
                log_source = np.log2(source_freqs)
                result[freq] = float(np.interp(log_freq, log_source, source_thresholds))
                result[freq] = round(result[freq] / 5) * 5  # Round to 5 dB

        return result

    def generate_population(
        self,
        n_per_phenotype: Optional[Dict[str, int]] = None,
        seed: int = 42
    ) -> List[Dict]:
        """
        Generate a population of audiograms across all phenotypes.

        Parameters
        ----------
        n_per_phenotype : dict, optional
            {phenotype_name: n}. If None, uses n_target from definitions.
        seed : int
            Random seed for reproducibility

        Returns
        -------
        list of dict
            Each dict contains: phenotype, audiogram, listener_id, cluster_id
        """
        rng = np.random.default_rng(seed)

        if n_per_phenotype is None:
            n_per_phenotype = {
                name: defn.n_target
                for name, defn in PHENOTYPE_DEFINITIONS.items()
            }

        population = []
        listener_id = 0

        for phenotype_name, n in n_per_phenotype.items():
            defn = PHENOTYPE_DEFINITIONS[phenotype_name]
            for _ in range(n):
                audiogram = self.generate_audiogram(phenotype_name, rng=rng)
                population.append({
                    'listener_id': listener_id,
                    'phenotype': phenotype_name,
                    'cluster_id': defn.cluster_id,
                    'category': defn.category,
                    'audiogram': audiogram,
                })
                listener_id += 1

        return population


# =============================================================================
# PSYCHOMETRIC PARAMETER GENERATOR
# =============================================================================

class PsychometricParameterGenerator:
    """
    Generate psychometric function parameters for simulated listeners.

    Uses literature-derived distributions as specified in the manuscript:
    - Slope (σ): LogNormal(log(8), 0.3), mean ≈ 8 dB
    - False positive (α): Beta(1.5, 28.5), mean = 0.05
    - False negative (β): Beta(1, 49), mean = 0.02
    """

    def __init__(self):
        # Literature-derived parameters (from manuscript)
        self.slope_mu = np.log(8)  # LogNormal location
        self.slope_sigma = 0.3     # LogNormal scale

        self.alpha_a = 1.5   # Beta shape for false positive
        self.alpha_b = 28.5

        self.beta_a = 1.0    # Beta shape for false negative
        self.beta_b = 49.0

    def generate(self, rng: Optional[np.random.Generator] = None) -> Dict[str, float]:
        """
        Generate psychometric parameters for a single listener.

        Returns
        -------
        dict with keys: slope, false_positive_rate, false_negative_rate
        """
        if rng is None:
            rng = np.random.default_rng()

        # Slope: LogNormal distribution, mean ≈ 8 dB
        slope = rng.lognormal(self.slope_mu, self.slope_sigma)
        slope = np.clip(slope, 3, 20)  # Reasonable bounds

        # False positive rate: Beta distribution, mean = 0.05
        alpha = rng.beta(self.alpha_a, self.alpha_b)
        alpha = np.clip(alpha, 0.01, 0.20)

        # False negative rate: Beta distribution, mean = 0.02
        beta = rng.beta(self.beta_a, self.beta_b)
        beta = np.clip(beta, 0.01, 0.15)

        return {
            'slope': float(slope),
            'false_positive_rate': float(alpha),
            'false_negative_rate': float(beta),
        }

    def generate_with_covariates(
        self,
        has_tinnitus: bool = False,
        tinnitus_freq: Optional[int] = None,
        has_cognitive_concerns: bool = False,
        is_first_audiogram: bool = False,
        rng: Optional[np.random.Generator] = None
    ) -> Dict[str, float]:
        """
        Generate parameters adjusted for covariates.

        As specified in manuscript:
        - Tinnitus: +0.05 to α at frequencies within ±1 octave of tinnitus pitch
        - Cognitive concerns: +0.03 to β
        - First audiogram: +2 dB to σ
        """
        params = self.generate(rng)

        if is_first_audiogram:
            params['slope'] += 2.0

        if has_cognitive_concerns:
            params['false_negative_rate'] = min(0.15, params['false_negative_rate'] + 0.03)

        # Tinnitus adjustment is frequency-specific, handled separately
        params['has_tinnitus'] = has_tinnitus
        params['tinnitus_freq'] = tinnitus_freq

        return params


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_phenotype_names() -> List[str]:
    """Return list of all phenotype names."""
    return list(PHENOTYPE_DEFINITIONS.keys())


def get_phenotype_categories() -> Dict[str, List[str]]:
    """Return phenotypes grouped by category."""
    categories = {}
    for name, defn in PHENOTYPE_DEFINITIONS.items():
        if defn.category not in categories:
            categories[defn.category] = []
        categories[defn.category].append(name)
    return categories


def get_phenotype_proportions() -> Dict[str, float]:
    """Return target proportions for each phenotype."""
    total = sum(p.n_target for p in PHENOTYPE_DEFINITIONS.values())
    return {name: defn.n_target / total for name, defn in PHENOTYPE_DEFINITIONS.items()}


def validate_gmm_models() -> Dict[str, bool]:
    """
    Check GMM model availability and status.

    Returns dict with model status information.
    """
    status = {
        'model_exists': GMM_MODEL_PATH.exists(),
        'sklearn_exists': GMM_SKLEARN_PATH.exists(),
        'is_approximate': USE_APPROXIMATE_GMM,
    }

    if status['model_exists']:
        try:
            gmm = get_gmm_model()
            gmm.load()
            status['n_components'] = gmm.n_components
            status['is_loaded'] = True
            status['model_is_approximate'] = gmm.is_approximate
        except Exception as e:
            status['is_loaded'] = False
            status['error'] = str(e)

    return status
