# Stage 1 Registered Report: Simulation Plan

## Overview

This document outlines the plan to revise and extend the in-silico simulation codebase to generate all results required for the Stage 1 manuscript submission.

---

## Current State vs Requirements

### What the Manuscript Requires

From `docs/from_manuscript_claude.md` and `main.tex`:

1. **n = 2,200 virtual listeners** across **11 phenotypes**
2. **Summary statistics**: Mean trials, efficiency gain (δN), test-retest variability (σ)
3. **Effect size validation**: δN = 8 trials, δσ = 2 dB, ρ = 0.6
4. **Key figures**:
   - Efficiency comparison (BSA vs Bayesian by phenotype)
   - Reliability comparison (test-retest distributions)
   - Phenotype-specific predictions (H3)
   - Posterior evolution examples
5. **H3 (Phenotype matching)**: Correlation between predicted and observed efficiency gains

### What Currently Exists

| Component | Status | Gap |
|-----------|--------|-----|
| Listener phenotypes | 4 types | Need 11 types |
| Sample size | Up to 200 | Need 2,200 |
| mHW procedure | Complete | None |
| Bayesian procedure | Complete (uniform priors) | Need NHANES priors |
| H1 (Efficiency) | Implemented | None |
| H2 (Reliability) | Implemented | Need ICC |
| H3 (Phenotype matching) | Not implemented | Full implementation needed |
| NHANES priors | Infrastructure only | Need integration |
| Visualization | 3 modes available | Need manuscript figures |

---

## Implementation Plan

### Phase 1: Extend Phenotype Definitions

**Task**: Create 11 phenotypes matching manuscript Table 1

```python
PHENOTYPES = {
    # Within normal limits
    'flat_normal': {'n': 330, 'pattern': 'flat', 'severity': 'normal'},

    # Age-related (presbycusis)
    'mild_hf_sloping': {'n': 264, 'pattern': 'sloping', 'severity': 'mild'},
    'moderate_hf_sloping': {'n': 220, 'pattern': 'sloping', 'severity': 'moderate'},
    'severe_hf_sloping': {'n': 176, 'pattern': 'sloping', 'severity': 'severe'},
    'ski_slope': {'n': 110, 'pattern': 'ski_slope', 'severity': 'severe'},

    # Noise-induced
    'notch_4k_mild': {'n': 220, 'pattern': '4k_notch', 'severity': 'mild'},
    'notch_4k_moderate': {'n': 176, 'pattern': '4k_notch', 'severity': 'moderate'},

    # Conductive
    'flat_mild_conductive': {'n': 176, 'pattern': 'flat', 'severity': 'mild_conductive'},
    'low_freq_ascending': {'n': 132, 'pattern': 'ascending', 'severity': 'mild'},

    # Mixed
    'cookie_bite': {'n': 176, 'pattern': 'cookie_bite', 'severity': 'moderate'},
    'reverse_slope': {'n': 220, 'pattern': 'reverse_slope', 'severity': 'moderate'},
}
# Total: 2,200
```

**Files to modify**:
- `tests/full_mhw_bayes_sim/full_mhw_bayes_sim.py` - ListenerGenerator class
- Create new config: `configs/stage1_manuscript.yaml`

### Phase 2: Integrate NHANES Priors into Bayesian Procedure

**Task**: Replace uniform priors with NHANES-derived KDE priors

**Current** (in `basic_bayes.py`):
```python
self.pdf = np.ones(self.n_points) / self.n_points  # Uniform
```

**Required**:
```python
from audiometry_ai.priors import get_threshold_prior
self.pdf = get_threshold_prior(age, sex, frequency, ear, priors_path)
```

**Files to modify**:
- `audiometry_ai/procedures/basic_bayes.py` - Add informed prior option
- `audiometry_ai/priors/__init__.py` - Export functions

### Phase 3: Implement Phenotype Matching (H3)

**Task**: Implement the phenotype matching algorithm for H3

**Algorithm** (from manuscript):
1. Compute audiogram features: slope, notch depth, asymmetry
2. Compute response features: false positive rate, variability
3. Compute efficiency: trials to threshold
4. Calculate Mahalanobis distance to phenotype centroids
5. Assign to minimum-distance phenotype

**New file**: `audiometry_ai/analysis/phenotype_matching.py`

```python
class PhenotypeMatching:
    def __init__(self, simulation_results):
        self.centroids = self._compute_centroids(simulation_results)

    def extract_features(self, participant):
        """Extract matching features from participant data."""
        return {
            'slope': self._compute_slope(participant.audiogram),
            'notch_depth': self._compute_notch(participant.audiogram),
            'asymmetry': self._compute_asymmetry(participant.audiogram),
            'fp_rate': participant.estimated_alpha,
            'response_variability': participant.response_std,
            'trials_mhw': participant.trials_mhw,
            'trials_bayes': participant.trials_bayes,
        }

    def match(self, features):
        """Return phenotype with minimum Mahalanobis distance."""
        distances = {}
        for phenotype, centroid in self.centroids.items():
            distances[phenotype] = mahalanobis(features, centroid)
        return min(distances, key=distances.get)
```

### Phase 4: Add ICC Calculation for H2

**Task**: Implement intraclass correlation coefficient for reliability

**New addition to** `audiometry_ai/analysis/`:

```python
def compute_icc(test1, test2, icc_type='ICC(2,1)'):
    """
    Compute intraclass correlation coefficient.

    ICC(2,1) - Two-way random effects, absolute agreement
    """
    from scipy import stats
    # Implementation using ANOVA decomposition
    ...
```

### Phase 5: Create Manuscript Figure Generation

**Task**: Create publication-ready figures for Stage 1

**Required figures**:

1. **Figure: Efficiency Comparison** (`fig_efficiency.pdf`)
   - Box/violin plot: trials by phenotype and procedure
   - Effect size annotations

2. **Figure: Reliability Comparison** (`fig_reliability.pdf`)
   - Bland-Altman plots for each procedure
   - Test-retest difference distributions

3. **Figure: Phenotype Predictions** (`fig_phenotype_predictions.pdf`)
   - Expected efficiency gain per phenotype
   - Error bars with 95% CI

4. **Figure: Posterior Evolution** (`fig_posterior.pdf`)
   - Already exists, may need refinement

**New file**: `audiometry_ai/visualization/manuscript_figures.py`

### Phase 6: Create Stage 1 Configuration

**New config**: `configs/stage1_manuscript.yaml`

```yaml
simulation:
  n_listeners: 2200
  n_repeats: 2
  random_seed: 42
  parallel_jobs: 8

phenotypes:
  flat_normal:
    proportion: 0.15  # 330/2200
    thresholds:
      pattern: flat
      range: [0, 20]
  mild_hf_sloping:
    proportion: 0.12  # 264/2200
    thresholds:
      pattern: sloping
      range: [10, 40]
  # ... all 11 phenotypes

psychometric:
  slope:
    distribution: lognormal
    params: {mu: 2.08, sigma: 0.3}  # mean ≈ 8 dB
  false_positive:
    distribution: beta
    params: {a: 1.5, b: 28.5}  # mean = 0.05
  false_negative:
    distribution: beta
    params: {a: 1, b: 49}  # mean = 0.02

procedures:
  mhw:
    starting_level: 40
    step_down: 10
    step_up: 5
    threshold_criterion: 0.5
  bayesian:
    use_nhanes_priors: true
    convergence_criterion: 5.0
    max_trials: 30

analysis:
  hypotheses:
    H1_efficiency: true
    H2_reliability: true
    H3_phenotype_matching: true
  alpha: 0.05
  effect_size_threshold: 0.5

output:
  figures:
    - efficiency_comparison
    - reliability_comparison
    - phenotype_predictions
    - posterior_evolution
  format: pdf
  dpi: 300
```

---

## File Changes Summary

### New Files

| File | Purpose |
|------|---------|
| `configs/stage1_manuscript.yaml` | Full Stage 1 configuration |
| `audiometry_ai/analysis/phenotype_matching.py` | H3 implementation |
| `audiometry_ai/analysis/icc.py` | ICC calculation |
| `audiometry_ai/visualization/manuscript_figures.py` | Publication figures |

### Modified Files

| File | Changes |
|------|---------|
| `audiometry_ai/procedures/basic_bayes.py` | Add NHANES prior support |
| `audiometry_ai/priors/__init__.py` | Export get_threshold_prior |
| `tests/full_mhw_bayes_sim/full_mhw_bayes_sim.py` | 11 phenotypes, H3 analysis |
| `audiometry_ai/priors/conditioning.py` | NHANES-validated effects (DONE) |

---

## Expected Outputs

### Summary Statistics

```
SIMULATION RESULTS (n=2,200, 2 repeats)
=========================================

H1: EFFICIENCY
--------------
mHW mean trials: XX.X (SD: X.X)
Bayesian mean trials: XX.X (SD: X.X)
Efficiency gain (δN): X.X trials [95% CI: X.X, X.X]
Effect size (Cohen's d): X.XX
p-value: < 0.001

H2: RELIABILITY
---------------
mHW test-retest σ: X.X dB
Bayesian test-retest σ: X.X dB
Reliability improvement (δσ): X.X dB [95% CI: X.X, X.X]
ICC (mHW): 0.XX
ICC (Bayesian): 0.XX

H3: PHENOTYPE MATCHING
----------------------
Predicted-observed correlation (ρ): 0.XX [95% CI: X.X, X.X]
```

### Phenotype-Specific Results

```
Phenotype              | n    | mHW trials | Bayes trials | Δ trials | Effect
-----------------------|------|------------|--------------|----------|--------
Flat normal            | 330  | XX.X       | XX.X         | X.X      | X.XX
Mild HF sloping        | 264  | XX.X       | XX.X         | X.X      | X.XX
Moderate HF sloping    | 220  | XX.X       | XX.X         | X.X      | X.XX
...
```

---

## Timeline

| Phase | Task | Estimated Time |
|-------|------|----------------|
| 1 | Extend phenotype definitions | 2-3 hours |
| 2 | Integrate NHANES priors | 3-4 hours |
| 3 | Implement phenotype matching (H3) | 4-5 hours |
| 4 | Add ICC calculation | 1-2 hours |
| 5 | Create manuscript figures | 3-4 hours |
| 6 | Create Stage 1 config and run | 2-3 hours |
| **Total** | | **15-21 hours** |

---

## Validation Checklist

Before submission, verify:

- [ ] n = 2,200 listeners simulated
- [ ] All 11 phenotypes represented with correct proportions
- [ ] Test-retest design (2 sessions per listener)
- [ ] NHANES priors integrated into Bayesian procedure
- [ ] H1 (Efficiency): δN calculated with 95% CI
- [ ] H2 (Reliability): ICC and σ calculated
- [ ] H3 (Phenotype matching): ρ correlation calculated
- [ ] All figures generated in PDF format (300 DPI)
- [ ] Results match expected effect sizes (δN ≈ 8, δσ ≈ 2, ρ ≈ 0.6)
- [ ] Code committed to GitHub
- [ ] OSF materials prepared

---

## Notes

1. **NHANES priors**: Already built and saved to `data/nhanes/priors/nhanes_priors.pkl`

2. **Conditioning effects**: Updated with NHANES-validated values (January 2025):
   - Sex: Increased HF effects (+10-14 dB at 3-8 kHz)
   - Diabetes: Reduced to +2 dB (not significant in NHANES)
   - CV risk: Low-frequency pattern validated (+8 dB at 500 Hz)

3. **Psychometric parameters**: Use literature-derived priors as specified in manuscript:
   - σ ~ LogNormal(log(8), 0.3)
   - α ~ Beta(1.5, 28.5)
   - β ~ Beta(1, 49)

---

*Plan created: January 2025*
*Status: Ready for implementation*
