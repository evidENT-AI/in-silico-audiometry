# GMM Phenotype Models

This directory contains Gaussian Mixture Models (GMM) for audiometric phenotype simulation.

## Current Status

**Mode**: Approximate GMM (placeholder)

The current models (`gmm_9cluster_rnent.pkl` and `gmm_9cluster_sklearn.pkl`) are approximate versions based on published literature values. They will be replaced with the exact GMM model from the RNENT clinical dataset collaboration.

## Model Files

| File | Description |
|------|-------------|
| `gmm_9cluster_rnent.pkl` | Dictionary format with means, covariances, weights |
| `gmm_9cluster_sklearn.pkl` | Sklearn GaussianMixture object for sampling |

## Integration Instructions

When the real GMM model from the RNENT collaboration is available:

### 1. Model Format Requirements

The model should be a Python dictionary with:

```python
{
    'weights': np.ndarray,      # Shape: (9,) - cluster weights summing to 1
    'means': np.ndarray,        # Shape: (9, 6) - mean audiogram per cluster
    'covariances': np.ndarray,  # Shape: (9, 6, 6) - covariance matrix per cluster
    'is_approximate': False,    # Set to False for real model
    'n_samples': int,           # Number of patients in training data
    'source': str,              # Data source identifier
    'frequencies': [250, 500, 1000, 2000, 4000, 8000],
}
```

### 2. Integration Steps

1. **Backup current models**:
   ```bash
   mv gmm_9cluster_rnent.pkl gmm_9cluster_rnent_approximate.pkl
   mv gmm_9cluster_sklearn.pkl gmm_9cluster_sklearn_approximate.pkl
   ```

2. **Place new model** as `gmm_9cluster_rnent.pkl`

3. **Update configuration** in `audiometry_ai/simulation/phenotypes.py`:
   ```python
   USE_APPROXIMATE_GMM = False
   ```

4. **Validate model**:
   ```bash
   python scripts/validate_gmm_model.py
   ```

5. **Run test simulation**:
   ```bash
   python scripts/run_stage1_simulation.py --mini
   ```

### 3. Creating sklearn model (optional)

If providing raw parameters, create sklearn model:

```python
from sklearn.mixture import GaussianMixture
import pickle

# Load your model parameters
model = pickle.load(open('gmm_9cluster_rnent.pkl', 'rb'))

# Create sklearn GMM
gmm = GaussianMixture(n_components=9, covariance_type='full')
gmm.weights_ = model['weights']
gmm.means_ = model['means']
gmm.covariances_ = model['covariances']
gmm.precisions_cholesky_ = np.linalg.cholesky(np.linalg.inv(model['covariances']))

# Save
pickle.dump(gmm, open('gmm_9cluster_sklearn.pkl', 'wb'))
```

## Phenotype Definitions

The 9 clusters map to clinical phenotypes:

| Cluster ID | Phenotype | Category | Proportion |
|------------|-----------|----------|------------|
| 0 | moderate_sloping | presbycusis | 11% |
| 1 | mild_sloping | presbycusis | 8% |
| 2 | severe_profound | severe | 3% |
| 3 | near_normal_mild_hf | mild | 12% |
| 4 | moderate_high_freq | noise_induced | 8% |
| 5 | moderate_severe | presbycusis | 10% |
| 6 | mild_hf_drop | presbycusis | 20% |
| 7 | ski_slope | presbycusis | 15% |
| 8 | normal_hearing | normal | 13% |

## Validation Criteria

A valid GMM model should:

1. Have 9 components (clusters)
2. Have means for 6 frequencies (250-8000 Hz)
3. Have positive-definite covariance matrices
4. Have weights summing to 1.0
5. Produce audiograms in valid range (-10 to 120 dB HL)

## References

- RNENT dataset: [Citation pending]
- GMM methodology: [Citation pending]
