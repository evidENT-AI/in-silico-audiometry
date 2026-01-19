#!/usr/bin/env python3
"""
Validate GMM Phenotype Model

This script validates the format and properties of a GMM model
for audiometric phenotype simulation.

Usage:
    python scripts/validate_gmm_model.py
    python scripts/validate_gmm_model.py --model path/to/model.pkl
"""

import argparse
import pickle
import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audiometry_ai.simulation.phenotypes import (
    GMM_MODEL_PATH,
    GMM_SKLEARN_PATH,
    FREQUENCIES,
    PHENOTYPE_DEFINITIONS,
)


def validate_gmm_model(model_path: Path) -> bool:
    """
    Validate a GMM model file.

    Parameters
    ----------
    model_path : Path
        Path to the pickle file containing the GMM model

    Returns
    -------
    bool
        True if validation passes
    """
    print(f"Validating GMM model: {model_path}")
    print("=" * 60)

    # Load model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print("[PASS] Model loaded successfully")
    except Exception as e:
        print(f"[FAIL] Failed to load model: {e}")
        return False

    errors = []
    warnings = []

    # Check model type
    if isinstance(model, dict):
        print("[INFO] Model is dictionary format")

        # Check required keys
        required_keys = ['weights', 'means', 'covariances']
        for key in required_keys:
            if key not in model:
                errors.append(f"Missing required key: {key}")
            else:
                print(f"[PASS] Found key: {key}")

        if errors:
            for e in errors:
                print(f"[FAIL] {e}")
            return False

        weights = model['weights']
        means = model['means']
        covariances = model['covariances']

        # Check dimensions
        n_components = len(weights)
        n_frequencies = len(FREQUENCIES)

        print(f"\n[INFO] Model dimensions:")
        print(f"  - Components (clusters): {n_components}")
        print(f"  - Frequencies: {means.shape[1] if len(means.shape) > 1 else 'N/A'}")

        # Validate weights
        if n_components != 9:
            warnings.append(f"Expected 9 components, got {n_components}")
        if not np.isclose(np.sum(weights), 1.0, atol=1e-6):
            errors.append(f"Weights should sum to 1.0, got {np.sum(weights)}")
        else:
            print(f"[PASS] Weights sum to {np.sum(weights):.6f}")

        if np.any(weights < 0):
            errors.append("Negative weights found")
        else:
            print("[PASS] All weights are non-negative")

        # Validate means
        expected_mean_shape = (n_components, n_frequencies)
        if means.shape != expected_mean_shape:
            warnings.append(f"Expected means shape {expected_mean_shape}, got {means.shape}")
        else:
            print(f"[PASS] Means shape: {means.shape}")

        if np.any(means < -10) or np.any(means > 120):
            warnings.append("Some mean values outside valid audiogram range (-10, 120)")
        else:
            print("[PASS] Mean values in valid range")

        # Validate covariances
        expected_cov_shape = (n_components, n_frequencies, n_frequencies)
        if covariances.shape != expected_cov_shape:
            errors.append(f"Expected covariances shape {expected_cov_shape}, got {covariances.shape}")
        else:
            print(f"[PASS] Covariances shape: {covariances.shape}")

        # Check positive definiteness
        for k in range(n_components):
            try:
                eigenvalues = np.linalg.eigvalsh(covariances[k])
                if np.any(eigenvalues <= 0):
                    errors.append(f"Covariance matrix for cluster {k} is not positive definite")
            except np.linalg.LinAlgError:
                errors.append(f"Failed to compute eigenvalues for cluster {k}")

        if not any('positive definite' in e for e in errors):
            print("[PASS] All covariance matrices are positive definite")

        # Check optional keys
        optional_keys = ['is_approximate', 'n_samples', 'source', 'frequencies']
        for key in optional_keys:
            if key in model:
                print(f"[INFO] Optional key '{key}': {model[key]}")

    else:
        # Assume sklearn GaussianMixture
        print("[INFO] Model appears to be sklearn GaussianMixture")
        try:
            n_components = model.n_components
            weights = model.weights_
            means = model.means_
            covariances = model.covariances_
            print(f"[PASS] Extracted model parameters")
            print(f"  - Components: {n_components}")
            print(f"  - Means shape: {means.shape}")
            print(f"  - Covariances shape: {covariances.shape}")
        except AttributeError as e:
            errors.append(f"Could not extract model parameters: {e}")

    # Test sampling
    print("\n[INFO] Testing sampling...")
    try:
        # For dictionary models, use GMMModel wrapper
        if isinstance(model, dict):
            from audiometry_ai.simulation.phenotypes import GMMModel
            gmm = GMMModel(model_path)
            gmm.load()
            audiograms, labels = gmm.sample(n_samples=100, random_state=42)
        else:
            # For sklearn models, sample directly
            audiograms, labels = model.sample(n_samples=100)
            audiograms = np.clip(audiograms, -10, 120)
            audiograms = np.round(audiograms / 5) * 5
        print(f"[PASS] Successfully sampled 100 audiograms")
        print(f"  - Audiogram shape: {audiograms.shape}")
        print(f"  - Label distribution: {np.bincount(labels, minlength=9)}")

        # Check audiogram validity
        if np.any(audiograms < -10) or np.any(audiograms > 120):
            warnings.append("Some sampled values outside valid range")
        else:
            print("[PASS] All sampled values in valid range")

        # Sample from specific cluster (only for dict models)
        if isinstance(model, dict):
            for cluster_id in [0, 4, 8]:
                cluster_samples = gmm.sample_from_cluster(cluster_id, n_samples=10)
                print(f"[PASS] Sampled from cluster {cluster_id}: shape {cluster_samples.shape}")
        else:
            print("[INFO] Skipping cluster-specific sampling for sklearn model")

    except Exception as e:
        errors.append(f"Sampling failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    if errors:
        print(f"\n[ERRORS] {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")

    if warnings:
        print(f"\n[WARNINGS] {len(warnings)} warning(s):")
        for w in warnings:
            print(f"  - {w}")

    if not errors:
        if warnings:
            print("\n[RESULT] PASSED with warnings")
        else:
            print("\n[RESULT] PASSED")
        return True
    else:
        print("\n[RESULT] FAILED")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Validate GMM phenotype model"
    )
    parser.add_argument(
        '--model', '-m', type=str, default=None,
        help=f'Path to model file (default: {GMM_MODEL_PATH})'
    )
    parser.add_argument(
        '--sklearn', action='store_true',
        help='Also validate sklearn model'
    )

    args = parser.parse_args()

    model_path = Path(args.model) if args.model else GMM_MODEL_PATH

    success = validate_gmm_model(model_path)

    if args.sklearn and GMM_SKLEARN_PATH.exists():
        print("\n" + "=" * 60)
        print("SKLEARN MODEL VALIDATION")
        print("=" * 60)
        sklearn_success = validate_gmm_model(GMM_SKLEARN_PATH)
        success = success and sklearn_success

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
