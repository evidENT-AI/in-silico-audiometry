# Manuscript Figure Descriptions

This document describes the figures generated from the Stage 1 in-silico simulation for the Registered Report.

---

## Figure 1: Population and Phenotype Overview

**File**: `fig1_population_overview.{png,pdf,svg}`

**Description**: Overview of the virtual listener population used in the simulation.

### Panels

- **Panel A**: Phenotype distribution showing the number of virtual listeners in each of the 9 GMM-derived hearing loss phenotype categories. Categories are color-coded by severity/type.

- **Panel B**: Example audiograms from each phenotype category, demonstrating the diversity of hearing loss configurations in the simulated population. X-axis: frequency (Hz, log scale), Y-axis: hearing level (dB HL, inverted).

- **Panel C**: Distribution of psychometric parameters across the population:
  - Left: Psychometric function slope (sigma, dB)
  - Center: False positive rate (alpha)
  - Right: False negative rate (beta)

**Relevance**: Demonstrates population diversity and validates that the simulation covers clinically relevant hearing profiles.

---

## Figure 2: Efficiency Results (H1)

**File**: `fig2_efficiency.{png,pdf,svg}`

**Description**: Comparison of testing efficiency between Bayesian and mHW procedures.

### Panels

- **Panel A**: Paired boxplot comparing total trial counts between mHW (red) and Bayesian (blue) procedures. Individual data points overlaid. Shows the primary H1 finding of ~24% trial reduction.

- **Panel B**: Trial reduction percentage by phenotype. Horizontal bar chart showing efficiency gains for each hearing loss phenotype, color-coded by category.

- **Panel C**: Accuracy vs. trials scatter plot. Shows mean absolute error (dB) vs. total trials for both procedures. Demonstrates that efficiency gains do not come at the cost of accuracy.

- **Panel D**: Effect size forest plot. Cohen's d with 95% CI for trial reduction at each test frequency (250-8000 Hz).

**Relevance**: Primary evidence for H1 hypothesis - Bayesian procedure is more efficient.

---

## Figure 3: Reliability Results (H2)

**File**: `fig3_reliability.{png,pdf,svg}`

**Description**: Test-retest reliability comparison between procedures.

### Panels

- **Panel A**: Bland-Altman plot for Bayesian procedure. Shows difference between test and retest thresholds vs. mean threshold. Horizontal lines indicate mean bias and limits of agreement (LoA).

- **Panel B**: Bland-Altman plot for mHW procedure. Same format as Panel A, allowing direct visual comparison of variability.

- **Panel C**: ICC comparison bar chart. Shows ICC(2,1) values with 95% CI for both procedures. Reference lines at 0.75 (good) and 0.9 (excellent) thresholds.

- **Panel D**: Test-retest scatter plot. Threshold from test session 1 vs. session 2 for both procedures. Identity line shown for reference.

**Relevance**: Primary evidence for H2 hypothesis - both procedures are reliable, with Bayesian showing tighter LoA.

---

## Figure 4: Phenotype Matching Results (H3)

**File**: `fig4_phenotype_matching.{png,pdf,svg}`

**Description**: Analysis of phenotype-based efficiency prediction.

### Panels

- **Panel A**: Predicted vs. observed efficiency gains scatter plot. Shows how well the phenotype-based model predicts individual efficiency gains. Identity and regression lines shown.

- **Panel B**: Confusion matrix for phenotype classification. Shows cross-validation accuracy of matching estimated audiograms to true phenotype categories.

- **Panel C**: Feature importance for phenotype matching. Horizontal bar chart showing relative importance of audiometric and procedural features.

- **Panel D**: Correlation coefficient with 95% CI. Visual representation of the predicted-observed correlation with interpretation thresholds.

**Relevance**: Evidence for H3 hypothesis - phenotype matching capability. **Note**: This requires proper GMM phenotypes and human data for full validation.

---

## Figure 5: Summary

**File**: `fig5_summary.{png,pdf,svg}`

**Description**: High-level summary of key findings across all hypotheses.

### Panels

- **Panel A**: Hypothesis summary table. Tabular presentation of key metrics and p-values for H1, H2, and H3.

- **Panel B**: Radar chart comparing Bayesian and mHW procedures across multiple dimensions:
  - Efficiency (trial reduction)
  - Accuracy (inverse error)
  - Reliability (ICC)
  - Consistency (inverse test-retest SD)
  - Matching (H3 correlation)

**Relevance**: At-a-glance comparison for manuscript abstract/discussion.

---

## Figure 6: Bayesian Hypothesis Testing

**File**: `fig6_bayesian.{png,pdf,svg}`

**Description**: Bayesian statistical analysis results as specified in the Registered Report analysis plan.

### Panels

- **Panel A**: Bayes Factors comparison. Bar chart (log scale) showing BF10 for H1 and H2 with interpretation threshold reference lines (moderate=3, strong=10, very strong=30, extreme=100).

- **Panel B**: H1 posterior distribution. Posterior density for efficiency gain (trials) with:
  - 95% HDI shaded region
  - Practical significance threshold (delta_min = 5 trials)
  - Posterior mean annotation

- **Panel C**: H2 posterior distribution. Posterior density for reliability improvement (dB) with:
  - 95% HDI shaded region
  - Practical significance threshold (delta_min = 1 dB)
  - Posterior mean annotation

- **Panel D**: Probability of practical significance. Bar chart showing P(delta > delta_min | data) for both hypotheses with evidence threshold reference lines.

**Relevance**: Core Bayesian analysis required by the Registered Report protocol.

---

## Technical Specifications

| Property | Value |
|----------|-------|
| Resolution | 300 DPI |
| Formats | PNG, PDF, SVG |
| Figure Width | 7.0 inches (double column) or 3.5 inches (single column) |
| Font | Arial |
| Color Scheme | Colorblind-friendly (blue/red for Bayesian/mHW) |

---

## Reproducing Figures

```bash
# Generate figures from existing results
python scripts/generate_manuscript_figures.py --results results/stage1_mini

# Generate figures with interactive display
python scripts/generate_manuscript_figures.py --results results/stage1_mini --show

# Custom output directory
python scripts/generate_manuscript_figures.py --results results/stage1_mini --output my_figures/
```

---

*Generated by in-silico-audiometry manuscript figure generation system*
