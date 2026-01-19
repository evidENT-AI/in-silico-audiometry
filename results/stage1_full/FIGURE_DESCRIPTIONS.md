# Figure Descriptions - Stage 1 Simulation

## Figure 1: Population Overview

**File:** `fig1_population_overview.{png,pdf,svg}`

**Caption:** Population characteristics and simulation parameters. **(A)** Distribution of listeners across audiometric phenotypes, with normal hearing comprising 55% of the sample (n=1,210). Colors indicate phenotype categories: green (normal), purple (presbycusis), orange (noise-induced), grey (severe/mild). **(B)** Representative audiograms from each phenotype category showing the characteristic patterns. Note the 4 kHz notch with 8 kHz recovery in the noise-induced (orange) phenotype. Legend positioned outside plot for clarity. **(C)** Distribution of psychometric function parameters across the simulated population: slope (σ, mean=8.4 dB), false positive rate (α, mean=0.050), and false negative rate (β, mean=0.022).

---

## Figure 2: Efficiency Results (H1)

**File:** `fig2_efficiency.{png,pdf,svg}`

**Caption:** Efficiency comparison between Bayesian and modified Hughson-Westlake (mHW) procedures. **(A)** Trial count distributions showing 38.0% reduction with the Bayesian procedure (mean: 46.7 vs 74.5 trials). Individual data points shown with jitter. **(B)** Trial reduction percentage by phenotype, color-coded by category. All phenotypes show >35% reduction. **(C)** Relationship between total trials and mean absolute error, demonstrating equivalent accuracy despite fewer trials with the Bayesian procedure. Both procedures achieve errors within the 5 dB clinical criterion (dashed line).

---

## Figure 3: Reliability Results (H2)

**File:** `fig3_reliability.{png,pdf,svg}`

**Caption:** Test-retest reliability comparison. **(A-B)** Bland-Altman difference plots for Bayesian and mHW procedures, respectively. Bias and 95% limits of agreement (LoA) shown in panel titles. The Bayesian procedure shows narrower LoA (±8.2 dB vs ±11.7 dB). **(C)** Intraclass correlation coefficients (ICC) with 95% confidence intervals. Both procedures achieve excellent reliability (>0.90), with the Bayesian procedure showing a small but significant advantage (0.986 vs 0.975). Y-axis zoomed to 0.90-1.00 range for visibility. **(D)** Test-retest scatter plot for the Bayesian procedure showing strong agreement along the identity line (dashed).

---

## Figure 4: Phenotype Matching Results (H3)

**File:** `fig4_phenotype_matching.{png,pdf,svg}`

**Caption:** Phenotype matching and efficiency prediction. **(A)** Predicted vs observed efficiency gains showing the negative correlation (r = -0.586, p < 0.0001) between phenotype-based predictions and actual trial reductions. The regression line (blue) demonstrates that greater predicted gains correspond to greater observed gains. Points colored by phenotype category. **(B)** Confusion matrix for phenotype classification showing 79.5% cross-validation accuracy. Diagonal elements indicate correct classifications. **(C)** Feature importance for phenotype classification, with response variability, efficiency gain, and audiogram asymmetry being the most predictive features. **(D)** Distribution of efficiency gains by phenotype category. Box plots show median and interquartile range, with colors indicating category. Normal hearing shows consistent moderate gains, while severe phenotypes show highest variability.

---

## Figure 5: Summary

**File:** `fig5_summary.{png,pdf,svg}`

**Caption:** Summary of hypothesis testing results. **(A)** Table summarizing key metrics for each hypothesis: H1 (37.3% trial reduction, p < 0.0001), H2 (ICC 0.986, p = 0.001), H3 (r = -0.586, p < 0.0001). **(B)** Radar plot comparing Bayesian (blue) and mHW (red) procedures across five dimensions: Accuracy, Efficiency, Matching, Consistency, and Reliability. The Bayesian procedure shows clear advantages in efficiency while maintaining comparable accuracy and reliability.

---

## Figure 6: Bayesian Posterior Distributions

**File:** `fig6_bayesian.{png,pdf,svg}`

**Caption:** Bayesian posterior distributions for procedure comparison. Posteriors estimated via Bayesian bootstrap (5,000 samples). **(A)** Posterior distributions of mean total trials for both procedures. The Bayesian procedure (blue, mean=46.7 trials) shows substantially fewer trials than mHW (red, mean=74.5 trials), with no overlap between posteriors indicating decisive evidence for the efficiency advantage. Dashed vertical lines indicate posterior means. **(B)** Posterior distributions of test-retest standard deviation for both procedures. The Bayesian procedure (blue, mean=4.20 dB) demonstrates lower test-retest variability than mHW (red, mean=6.00 dB), with clear separation between posteriors supporting superior reliability. Both panels show histogram representations of MCMC-like samples rather than smooth approximations, reflecting the empirical uncertainty in the estimates.

---

## Technical Specifications

- **Format:** PNG (300 DPI), PDF (vector), SVG (vector)
- **Color scheme:** Colorblind-friendly (blue=#2166AC, red=#B2182B)
- **Font:** Arial
- **Style:** Nature formatting guidelines

## Data Availability

All figures are generated from the simulation results stored in:
- `stage1_summary.json` - Summary statistics
- `stage1_full_results.pkl` - Complete results including individual listener data

Regenerate figures with:
```bash
python scripts/generate_manuscript_figures.py --results results/stage1_full
```
