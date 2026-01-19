"""
Phenotype matching algorithm for H3 hypothesis testing.

This module implements the phenotype matching algorithm described in the
Stage 1 manuscript. The algorithm matches human participants to their
most likely simulated phenotype based on response characteristics.

Key features:
1. Extract audiogram features (slope, notch depth, asymmetry)
2. Extract response features (false positive rate, variability)
3. Compute efficiency metrics (trials to threshold)
4. Match to simulated phenotype centroids using Mahalanobis distance
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.spatial.distance import mahalanobis
from scipy import stats
import warnings


@dataclass
class PhenotypeFeatures:
    """Features extracted from a listener for phenotype matching."""
    # Audiogram features
    slope_low: float  # Slope 250-1000 Hz
    slope_high: float  # Slope 1000-8000 Hz
    notch_depth: float  # 4 kHz notch depth (if present)
    asymmetry: float  # Left-right asymmetry
    pta_4freq: float  # 4-frequency PTA (500, 1000, 2000, 4000 Hz)

    # Response features
    estimated_fp_rate: float  # Estimated false positive rate
    response_variability: float  # Within-session response variability

    # Efficiency features
    trials_mhw: int  # Total trials for mHW procedure
    trials_bayes: int  # Total trials for Bayesian procedure
    efficiency_gain: float  # trials_mhw - trials_bayes


@dataclass
class PhenotypeCentroid:
    """Centroid and covariance for a phenotype cluster."""
    name: str
    mean: np.ndarray  # Mean feature vector
    cov: np.ndarray  # Covariance matrix
    cov_inv: np.ndarray  # Inverse covariance (for Mahalanobis)
    n_samples: int


class PhenotypeMatching:
    """
    Match participants to simulated phenotypes based on response characteristics.

    This implements H3: testing whether predicted efficiency gains from
    simulations correlate with observed gains at the individual level.
    """

    # Feature names in order
    FEATURE_NAMES = [
        'slope_low', 'slope_high', 'notch_depth', 'asymmetry', 'pta_4freq',
        'estimated_fp_rate', 'response_variability',
        'trials_mhw', 'trials_bayes', 'efficiency_gain'
    ]

    def __init__(
        self,
        use_audiogram_features: bool = True,
        use_response_features: bool = True,
        use_efficiency_features: bool = True,
        regularization: float = 1e-6
    ):
        """
        Initialize phenotype matching.

        Parameters
        ----------
        use_audiogram_features : bool
            Include audiogram shape features in matching
        use_response_features : bool
            Include response pattern features in matching
        use_efficiency_features : bool
            Include efficiency metrics in matching
        regularization : float
            Regularization for covariance matrix inversion
        """
        self.use_audiogram = use_audiogram_features
        self.use_response = use_response_features
        self.use_efficiency = use_efficiency_features
        self.regularization = regularization

        self.centroids: Dict[str, PhenotypeCentroid] = {}
        self.feature_mask: np.ndarray = self._get_feature_mask()

    def _get_feature_mask(self) -> np.ndarray:
        """Get mask for which features to use in matching."""
        mask = []
        # Audiogram features (5)
        mask.extend([self.use_audiogram] * 5)
        # Response features (2)
        mask.extend([self.use_response] * 2)
        # Efficiency features (3)
        mask.extend([self.use_efficiency] * 3)
        return np.array(mask)

    def extract_features(
        self,
        audiogram: Dict[int, float],
        audiogram_left: Optional[Dict[int, float]] = None,
        response_data: Optional[Dict] = None,
        mhw_results: Optional[Dict] = None,
        bayes_results: Optional[Dict] = None
    ) -> PhenotypeFeatures:
        """
        Extract matching features from participant data.

        Parameters
        ----------
        audiogram : dict
            {frequency: threshold_dB} for right ear (or single ear)
        audiogram_left : dict, optional
            {frequency: threshold_dB} for left ear
        response_data : dict, optional
            Response pattern data for FP rate estimation
        mhw_results : dict, optional
            mHW procedure results
        bayes_results : dict, optional
            Bayesian procedure results

        Returns
        -------
        PhenotypeFeatures
            Extracted feature set
        """
        # Extract audiogram features
        slope_low = self._compute_slope(audiogram, 250, 1000)
        slope_high = self._compute_slope(audiogram, 1000, 8000)
        notch_depth = self._compute_notch_depth(audiogram)
        asymmetry = self._compute_asymmetry(audiogram, audiogram_left)
        pta_4freq = self._compute_pta(audiogram, [500, 1000, 2000, 4000])

        # Extract response features
        if response_data:
            estimated_fp_rate = response_data.get('estimated_fp_rate', 0.05)
            response_variability = response_data.get('response_variability', 0.1)
        else:
            estimated_fp_rate = 0.05
            response_variability = 0.1

        # Extract efficiency features
        if mhw_results:
            trials_mhw = sum(mhw_results.get('trial_counts', {}).values())
        else:
            trials_mhw = 0

        if bayes_results:
            trials_bayes = sum(
                len(prog) for prog in bayes_results.get('progression_patterns', {}).values()
            )
        else:
            trials_bayes = 0

        efficiency_gain = trials_mhw - trials_bayes

        return PhenotypeFeatures(
            slope_low=slope_low,
            slope_high=slope_high,
            notch_depth=notch_depth,
            asymmetry=asymmetry,
            pta_4freq=pta_4freq,
            estimated_fp_rate=estimated_fp_rate,
            response_variability=response_variability,
            trials_mhw=trials_mhw,
            trials_bayes=trials_bayes,
            efficiency_gain=efficiency_gain
        )

    def _compute_slope(
        self,
        audiogram: Dict[int, float],
        freq_low: int,
        freq_high: int
    ) -> float:
        """Compute audiogram slope between two frequencies (dB/octave)."""
        freqs = sorted([f for f in audiogram.keys() if freq_low <= f <= freq_high])
        if len(freqs) < 2:
            return 0.0

        thresholds = [audiogram[f] for f in freqs]
        log_freqs = np.log2(freqs)

        # Linear regression for slope
        slope, _, _, _, _ = stats.linregress(log_freqs, thresholds)
        return slope  # dB per octave

    def _compute_notch_depth(self, audiogram: Dict[int, float]) -> float:
        """
        Compute 4 kHz notch depth.

        Notch depth = threshold at 4kHz - average of 2kHz and 8kHz
        Positive value indicates notch.
        """
        if 4000 not in audiogram:
            return 0.0

        thresh_4k = audiogram[4000]

        # Get adjacent frequencies
        neighbors = []
        if 2000 in audiogram:
            neighbors.append(audiogram[2000])
        if 3000 in audiogram:
            neighbors.append(audiogram[3000])
        if 6000 in audiogram:
            neighbors.append(audiogram[6000])
        if 8000 in audiogram:
            neighbors.append(audiogram[8000])

        if len(neighbors) < 2:
            return 0.0

        avg_neighbors = np.mean(neighbors)
        return thresh_4k - avg_neighbors  # Positive = notch

    def _compute_asymmetry(
        self,
        audiogram_right: Dict[int, float],
        audiogram_left: Optional[Dict[int, float]]
    ) -> float:
        """Compute interaural asymmetry (average |right - left|)."""
        if audiogram_left is None:
            return 0.0

        common_freqs = set(audiogram_right.keys()) & set(audiogram_left.keys())
        if not common_freqs:
            return 0.0

        diffs = [abs(audiogram_right[f] - audiogram_left[f]) for f in common_freqs]
        return np.mean(diffs)

    def _compute_pta(
        self,
        audiogram: Dict[int, float],
        frequencies: List[int]
    ) -> float:
        """Compute pure-tone average for specified frequencies."""
        available = [audiogram[f] for f in frequencies if f in audiogram]
        if not available:
            return 0.0
        return np.mean(available)

    def features_to_vector(self, features: PhenotypeFeatures) -> np.ndarray:
        """Convert PhenotypeFeatures to numpy vector."""
        return np.array([
            features.slope_low,
            features.slope_high,
            features.notch_depth,
            features.asymmetry,
            features.pta_4freq,
            features.estimated_fp_rate,
            features.response_variability,
            features.trials_mhw,
            features.trials_bayes,
            features.efficiency_gain
        ])

    def fit_centroids(
        self,
        simulation_results: List[Dict],
        phenotype_key: str = 'phenotype'
    ) -> None:
        """
        Compute phenotype centroids from simulation results.

        Parameters
        ----------
        simulation_results : list of dict
            Simulation results with phenotype labels and features
        phenotype_key : str
            Key for phenotype label in results
        """
        # Group by phenotype
        by_phenotype: Dict[str, List[np.ndarray]] = {}

        for result in simulation_results:
            phenotype = result.get(phenotype_key, 'unknown')
            features = self.extract_features(
                audiogram=result.get('true_thresholds', result.get('audiogram', {})),
                mhw_results=result.get('mhw_results'),
                bayes_results=result.get('bayes_results')
            )
            vector = self.features_to_vector(features)

            if phenotype not in by_phenotype:
                by_phenotype[phenotype] = []
            by_phenotype[phenotype].append(vector)

        # Compute centroids
        for phenotype, vectors in by_phenotype.items():
            vectors = np.array(vectors)

            # Apply feature mask
            vectors_masked = vectors[:, self.feature_mask]
            n_samples = len(vectors)
            n_features = vectors_masked.shape[1]

            mean = np.mean(vectors_masked, axis=0)

            # Handle edge cases for covariance computation
            if n_samples < 2:
                # With only one sample, use identity matrix scaled by regularization
                cov = self.regularization * np.eye(n_features)
            else:
                cov = np.cov(vectors_masked, rowvar=False)
                # Ensure cov is 2D (np.cov returns scalar for 1D input)
                if cov.ndim == 0:
                    cov = np.array([[float(cov)]])
                elif cov.ndim == 1:
                    cov = np.diag(cov)
                # Regularize covariance for numerical stability
                cov = cov + self.regularization * np.eye(cov.shape[0])

            try:
                cov_inv = np.linalg.inv(cov)
            except np.linalg.LinAlgError:
                warnings.warn(f"Singular covariance for {phenotype}, using pseudoinverse")
                cov_inv = np.linalg.pinv(cov)

            self.centroids[phenotype] = PhenotypeCentroid(
                name=phenotype,
                mean=mean,
                cov=cov,
                cov_inv=cov_inv,
                n_samples=len(vectors)
            )

    def match(self, features: PhenotypeFeatures) -> Tuple[str, Dict[str, float]]:
        """
        Match features to the most likely phenotype.

        Parameters
        ----------
        features : PhenotypeFeatures
            Features to match

        Returns
        -------
        tuple
            (best_phenotype, distances_dict)
        """
        if not self.centroids:
            raise ValueError("No centroids fitted. Call fit_centroids() first.")

        vector = self.features_to_vector(features)
        vector_masked = vector[self.feature_mask]

        distances = {}
        for name, centroid in self.centroids.items():
            try:
                dist = mahalanobis(vector_masked, centroid.mean, centroid.cov_inv)
            except Exception:
                # Fallback to Euclidean
                dist = np.linalg.norm(vector_masked - centroid.mean)
            distances[name] = dist

        best = min(distances, key=distances.get)
        return best, distances

    def get_predicted_efficiency_gain(self, phenotype: str) -> float:
        """Get mean efficiency gain for a phenotype from fitted centroids."""
        if phenotype not in self.centroids:
            return 0.0

        centroid = self.centroids[phenotype]
        # Efficiency gain is the last feature in the masked set if included
        if self.use_efficiency:
            # Find index of efficiency_gain in masked features
            idx = sum(self.feature_mask[:9]) - 1  # Index 9 is efficiency_gain
            return centroid.mean[idx]
        return 0.0

    def compute_h3_correlation(
        self,
        participant_results: List[Dict],
        phenotype_key: str = 'matched_phenotype'
    ) -> Tuple[float, float, float]:
        """
        Compute H3 correlation: predicted vs observed efficiency gains.

        Parameters
        ----------
        participant_results : list of dict
            Results with matched phenotypes and observed efficiency

        Returns
        -------
        tuple
            (correlation, p_value, 95% CI lower, 95% CI upper)
        """
        predicted = []
        observed = []

        for result in participant_results:
            phenotype = result.get(phenotype_key)
            if phenotype is None:
                continue

            pred_gain = self.get_predicted_efficiency_gain(phenotype)
            obs_gain = result.get('efficiency_gain', 0)

            predicted.append(pred_gain)
            observed.append(obs_gain)

        if len(predicted) < 3:
            return 0.0, 1.0, -1.0, 1.0

        # Pearson correlation
        r, p = stats.pearsonr(predicted, observed)

        # 95% CI using Fisher z-transform
        n = len(predicted)
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_low, z_high = z - 1.96 * se, z + 1.96 * se
        ci_low, ci_high = np.tanh(z_low), np.tanh(z_high)

        return r, p, ci_low, ci_high


def cross_validate_matching(
    simulation_results: List[Dict],
    phenotype_key: str = 'phenotype',
    n_folds: int = 5,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Leave-one-out or k-fold cross-validation of phenotype matching.

    Returns accuracy metrics for the matching algorithm.
    """
    rng = np.random.default_rng(random_state)
    indices = np.arange(len(simulation_results))
    rng.shuffle(indices)

    fold_size = len(indices) // n_folds
    correct = 0
    total = 0

    for fold in range(n_folds):
        # Split
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else len(indices)
        test_idx = indices[test_start:test_end]
        train_idx = np.concatenate([indices[:test_start], indices[test_end:]])

        # Train
        train_data = [simulation_results[i] for i in train_idx]
        matcher = PhenotypeMatching()
        matcher.fit_centroids(train_data, phenotype_key)

        # Test
        for i in test_idx:
            result = simulation_results[i]
            true_phenotype = result.get(phenotype_key)

            features = matcher.extract_features(
                audiogram=result.get('true_thresholds', result.get('audiogram', {})),
                mhw_results=result.get('mhw_results'),
                bayes_results=result.get('bayes_results')
            )

            predicted, _ = matcher.match(features)
            if predicted == true_phenotype:
                correct += 1
            total += 1

    return {
        'accuracy': correct / total if total > 0 else 0,
        'n_correct': correct,
        'n_total': total,
        'n_folds': n_folds
    }
