# Stage 1 Simulation Results Summary

**Date:** January 2026
**Simulation:** Full 2200 listener Stage 1 Registered Report simulation
**Distribution:** `normal_majority` (55% normal hearing)
**Priors:** NHANES with full three-level conditioning

---

## Executive Summary

All three hypotheses are **strongly supported** by the simulation results:

| Hypothesis | Result | Evidence |
|------------|--------|----------|
| **H1: Efficiency** | Bayesian uses **37.3% fewer trials** | BF₁₀ = ∞ (Extreme) |
| **H2: Reliability** | Bayesian has **higher ICC** (0.986 vs 0.975) | BF₁₀ > 10⁶⁷ (Extreme) |
| **H3: Phenotype Matching** | **79.5% accuracy**, r = -0.59 | p < 10⁻²⁰⁰ |

---

## Population Characteristics

### Sample Size and Distribution
- **Total listeners:** 2,200
- **Test sessions:** 2 (test-retest design)
- **Total simulated audiograms:** 4,400

| Phenotype | Category | n | % |
|-----------|----------|---|---|
| normal_hearing | normal | 1,210 | 55.0% |
| near_normal_mild_hf | mild | 220 | 10.0% |
| mild_hf_drop | presbycusis | 220 | 10.0% |
| moderate_sloping | presbycusis | 110 | 5.0% |
| mild_sloping | presbycusis | 110 | 5.0% |
| moderate_high_freq | noise_induced | 110 | 5.0% |
| ski_slope | presbycusis | 110 | 5.0% |
| moderate_severe | presbycusis | 66 | 3.0% |
| severe_profound | severe | 44 | 2.0% |

### Demographics (NHANES-based)
- **Age:** 45.6 ± 18.1 years (range: 18-90)
- **Sex:** 44.9% male, 55.1% female

### Risk Factors (Level 2 Prior Conditioning)
- **Diabetes:** 10.7%
- **Cardiovascular risk:** 24.5%
- **Noise exposure:** 20.5%

---

## H1: Efficiency (Trial Count Reduction)

### Primary Outcome
The Bayesian procedure requires significantly fewer trials than modified Hughson-Westlake (mHW).

| Metric | Bayesian | mHW | Difference |
|--------|----------|-----|------------|
| Mean trials | 46.7 ± 9.0 | 74.5 ± 7.0 | **-27.8 trials** |
| Reduction | — | — | **37.3%** |

### Statistical Analysis

**Frequentist:**
- Paired t-test: t = 98.09
- p-value: < 10⁻³⁰⁰ (effectively 0)
- Cohen's d = **3.44** (very large effect)

**Bayesian:**
- Bayes Factor (BF₁₀): **∞** (Extreme evidence for H1)
- Posterior mean reduction: 27.79 trials
- 95% HDI: [27.21, 28.32] trials
- P(reduction > 5 trials): **100%**

### Accuracy Comparison
Both procedures achieve similar accuracy:
- Bayesian mean error: 2.73 dB
- mHW mean error: 2.90 dB

---

## H2: Test-Retest Reliability

### Primary Outcome
The Bayesian procedure demonstrates superior test-retest reliability.

| Metric | Bayesian | mHW |
|--------|----------|-----|
| ICC | **0.986** [0.985, 0.986] | 0.975 [0.974, 0.975] |
| Test-retest SD | **4.18 dB** | 5.96 dB |
| Bias | -0.03 dB | -0.002 dB |
| 95% LoA | ±8.2 dB | ±11.7 dB |

### Statistical Analysis

**Frequentist:**
- ICC difference: +0.011 (Bayesian advantage)
- Both ICCs in "excellent" range (>0.90)

**Bayesian:**
- Bayes Factor (BF₁₀): **1.39 × 10⁶⁷** (Extreme evidence for H1)
- Posterior mean improvement: 1.79 dB (in test-retest SD)
- 95% HDI: [1.55, 2.02] dB
- P(improvement > 1 dB): **100%**

**Bayesian ICC Estimates:**
- Bayesian procedure: 0.993 [0.992, 0.994], P(ICC > 0.9) = 100%
- mHW procedure: 0.987 [0.987, 0.988], P(ICC > 0.9) = 100%

---

## H3: Phenotype Matching

### Primary Outcome
Audiometric phenotype can be accurately predicted and correlates with efficiency gains.

| Metric | Value |
|--------|-------|
| Matching accuracy | **79.5%** (1,750/2,200) |
| Correlation (r) | **-0.586** |
| 95% CI | [-0.613, -0.558] |
| p-value | 2.69 × 10⁻²⁰³ |

### Interpretation
- The negative correlation indicates that phenotypes with **greater predicted efficiency gains** show **greater observed efficiency gains**
- Cross-validation accuracy of 79.5% demonstrates robust phenotype classification
- This supports the hypothesis that audiometric phenotype influences procedural efficiency

---

## Configuration

### Simulation Parameters
```yaml
n_listeners: 2200
seed: 42
distribution_preset: normal_majority
use_nhanes_priors: true
n_workers: 10
```

### Prior Conditioning (Three-Level Hierarchy)
1. **Level 1:** Age-sex stratification (NHANES empirical)
2. **Level 2:** Risk factor adjustments
   - Diabetes (NHANES-validated, reduced effects)
   - Cardiovascular risk (low-frequency emphasis validated)
   - Noise exposure (4 kHz notch pattern)
   - Tinnitus, ototoxicity, Meniere's (literature-based)
3. **Level 3:** Tympanometry (clinical-only, not in simulation)

### Phenotype Model
- 9-cluster GMM derived from RNENT clinical dataset
- NIHL phenotype updated with characteristic 4 kHz notch and 8 kHz recovery
- Approximate covariances estimated from cluster means

---

## Output Files

| File | Description |
|------|-------------|
| `stage1_summary.json` | Summary statistics (JSON) |
| `stage1_full_results.pkl` | Complete results (pickle) |
| `simulation.log` | Full console output |
| `RESULTS_SUMMARY.md` | This document |

---

## Conclusions

1. **H1 Supported:** The Bayesian procedure achieves a **37.3% reduction in trials** with equivalent accuracy, representing a clinically meaningful efficiency improvement.

2. **H2 Supported:** The Bayesian procedure demonstrates **superior test-retest reliability** (ICC 0.986 vs 0.975), with narrower limits of agreement (±8.2 dB vs ±11.7 dB).

3. **H3 Supported:** Audiometric phenotype matching achieves **79.5% accuracy** with a strong correlation (r = -0.59) between predicted and observed efficiency gains.

All results show **extreme Bayesian evidence** (BF₁₀ > 100) in favor of the alternative hypotheses, supporting the superiority of the Bayesian pure-tone audiometry procedure.

---

*Generated: January 2026*
*Simulation framework: in-silico-audiometry v1.0*
