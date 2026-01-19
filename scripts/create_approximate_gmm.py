"""
Create approximate GMM model from RNENT notebook cluster means.

This script creates a placeholder GMM model using the cluster means
extracted from the Audiogram_GMM_RNENT notebook. The covariances are
estimated based on typical audiometric variability.

Once the collaborator provides the actual fitted model, replace
data/phenotype_gmms/gmm_9cluster_rnent.pkl with their version.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
import pickle
from pathlib import Path

# Cluster means from gmm_scientific_reports.ipynb (cell 41)
# Format: [250Hz, 500Hz, 1000Hz, 2000Hz, 4000Hz, 8000Hz]
CLUSTER_MEANS = np.array([
    [40.87, 44.86, 50.11, 55.34, 64.47, 77.77],   # Cluster 1: Moderate sloping
    [33.96, 37.22, 41.44, 41.45, 47.70, 56.48],   # Cluster 2: Mild sloping
    [80.82, 92.55, 106.15, 116.01, 120.00, 107.82], # Cluster 3: Severe/profound
    [15.97, 15.57, 15.25, 15.63, 24.73, 38.22],   # Cluster 4: Near-normal
    [24.40, 26.66, 32.99, 47.02, 61.77, 74.11],   # Cluster 5: Moderate HF
    [67.92, 69.71, 72.25, 73.85, 83.32, 94.27],   # Cluster 6: Moderate-severe
    [15.67, 17.27, 20.16, 27.89, 48.19, 58.36],   # Cluster 7: Mild + HF drop
    [42.19, 48.46, 60.40, 75.51, 95.68, 101.35],  # Cluster 8: Steeply sloping
    [10.08, 9.06, 8.75, 7.94, 10.49, 12.33],      # Cluster 9: Normal hearing
])

# Cluster descriptions for reference
CLUSTER_DESCRIPTIONS = {
    0: "Moderate sloping (presbycusis)",
    1: "Mild sloping",
    2: "Severe/profound SNHL",
    3: "Near-normal with mild HF loss",
    4: "Moderate high-frequency loss",
    5: "Moderate-severe flat/sloping",
    6: "Mild with high-frequency drop",
    7: "Steeply sloping (ski-slope)",
    8: "Normal hearing",
}

# Approximate cluster proportions from notebook (visual estimate from bar chart)
# These sum to 1.0
CLUSTER_WEIGHTS = np.array([
    0.11,  # Cluster 1
    0.08,  # Cluster 2
    0.03,  # Cluster 3
    0.12,  # Cluster 4
    0.08,  # Cluster 5
    0.10,  # Cluster 6
    0.20,  # Cluster 7
    0.15,  # Cluster 8
    0.13,  # Cluster 9
])
CLUSTER_WEIGHTS = CLUSTER_WEIGHTS / CLUSTER_WEIGHTS.sum()  # Normalize

# Frequencies
FREQUENCIES = [250, 500, 1000, 2000, 4000, 8000]


def estimate_covariances(means: np.ndarray, base_variance: float = 100.0) -> np.ndarray:
    """
    Estimate covariance matrices for each cluster.

    Uses a model where:
    - Variance increases with threshold level (higher thresholds = more variability)
    - Adjacent frequencies are correlated (audiograms are smooth)
    - Correlation decreases with frequency distance

    Parameters
    ----------
    means : np.ndarray
        Cluster means, shape (n_clusters, n_features)
    base_variance : float
        Base variance at 0 dB threshold

    Returns
    -------
    np.ndarray
        Covariance matrices, shape (n_clusters, n_features, n_features)
    """
    n_clusters, n_features = means.shape
    covariances = np.zeros((n_clusters, n_features, n_features))

    for k in range(n_clusters):
        # Variance scales with threshold level
        # Higher thresholds typically have more variability
        variances = base_variance + 0.5 * means[k]  # Increases with threshold
        variances = np.clip(variances, 25, 400)  # Reasonable bounds (5-20 dB SD)

        # Build correlation matrix
        # Adjacent frequencies are highly correlated
        correlation = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(n_features):
                # Correlation decays with frequency distance
                octave_distance = abs(np.log2(FREQUENCIES[i] / FREQUENCIES[j]))
                correlation[i, j] = np.exp(-0.5 * octave_distance)

        # Convert to covariance matrix
        std_devs = np.sqrt(variances)
        covariances[k] = np.outer(std_devs, std_devs) * correlation

        # Add small regularization for numerical stability
        covariances[k] += 0.01 * np.eye(n_features)

    return covariances


def create_approximate_gmm() -> dict:
    """
    Create an approximate GMM model structure.

    Returns a dictionary that can be used similarly to a fitted
    sklearn GaussianMixture model.
    """
    covariances = estimate_covariances(CLUSTER_MEANS)

    # Compute precision matrices (inverse covariances)
    precisions = np.array([np.linalg.inv(cov) for cov in covariances])

    # Compute precision Cholesky (for sampling)
    precisions_chol = np.array([np.linalg.cholesky(prec) for prec in precisions])

    return {
        'n_components': 9,
        'means': CLUSTER_MEANS,
        'covariances': covariances,
        'weights': CLUSTER_WEIGHTS,
        'precisions': precisions,
        'precisions_cholesky': precisions_chol,
        'frequencies': FREQUENCIES,
        'descriptions': CLUSTER_DESCRIPTIONS,
        'covariance_type': 'full',
        'is_approximate': True,  # Flag that this is not the real model
        'source': 'Approximated from Audiogram_GMM_RNENT notebook means',
    }


def create_fitted_gmm() -> GaussianMixture:
    """
    Create a sklearn GaussianMixture object with the approximate parameters.

    This allows using standard sklearn methods like .sample() and .predict().
    """
    gmm = GaussianMixture(
        n_components=9,
        covariance_type='full',
        random_state=42,
    )

    # Manually set the fitted parameters
    gmm.weights_ = CLUSTER_WEIGHTS
    gmm.means_ = CLUSTER_MEANS
    gmm.covariances_ = estimate_covariances(CLUSTER_MEANS)
    gmm.precisions_cholesky_ = np.array([
        np.linalg.cholesky(np.linalg.inv(cov))
        for cov in gmm.covariances_
    ])

    # Mark as fitted
    gmm.converged_ = True
    gmm.n_iter_ = 1
    gmm.lower_bound_ = 0.0

    return gmm


def sample_audiograms(n_samples: int, random_state: int = None) -> tuple:
    """
    Sample audiograms from the approximate GMM.

    Parameters
    ----------
    n_samples : int
        Number of audiograms to sample
    random_state : int, optional
        Random seed

    Returns
    -------
    tuple
        (audiograms, cluster_labels) where audiograms is shape (n_samples, 6)
    """
    rng = np.random.default_rng(random_state)

    # Sample cluster assignments based on weights
    clusters = rng.choice(9, size=n_samples, p=CLUSTER_WEIGHTS)

    # Sample from each cluster's distribution
    covariances = estimate_covariances(CLUSTER_MEANS)
    audiograms = np.zeros((n_samples, 6))

    for i, k in enumerate(clusters):
        audiograms[i] = rng.multivariate_normal(CLUSTER_MEANS[k], covariances[k])

    # Clip to valid range and round to 5 dB steps
    audiograms = np.clip(audiograms, -10, 120)
    audiograms = np.round(audiograms / 5) * 5

    return audiograms, clusters


def main():
    """Create and save the approximate GMM model."""
    output_dir = Path(__file__).parent.parent / "data" / "phenotype_gmms"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model dictionary
    model_dict = create_approximate_gmm()

    # Save as pickle
    output_path = output_dir / "gmm_9cluster_rnent.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(model_dict, f)
    print(f"Saved approximate GMM model to: {output_path}")

    # Also save the fitted sklearn GMM for direct use
    gmm = create_fitted_gmm()
    sklearn_path = output_dir / "gmm_9cluster_sklearn.pkl"
    with open(sklearn_path, 'wb') as f:
        pickle.dump(gmm, f)
    print(f"Saved sklearn GMM to: {sklearn_path}")

    # Test sampling
    print("\nTesting sampling...")
    audiograms, labels = sample_audiograms(10, random_state=42)
    print(f"Sampled {len(audiograms)} audiograms")
    print(f"Cluster distribution: {np.bincount(labels, minlength=9)}")
    print(f"\nExample audiogram (cluster {labels[0]}):")
    print(f"  Frequencies: {FREQUENCIES}")
    print(f"  Thresholds:  {audiograms[0].astype(int).tolist()}")

    # Print cluster summary
    print("\n" + "="*60)
    print("GMM CLUSTER SUMMARY (9 clusters from RNENT)")
    print("="*60)
    for k in range(9):
        print(f"\nCluster {k+1}: {CLUSTER_DESCRIPTIONS[k]}")
        print(f"  Weight: {CLUSTER_WEIGHTS[k]:.1%}")
        print(f"  Means:  {CLUSTER_MEANS[k].astype(int).tolist()} dB")


if __name__ == "__main__":
    main()
