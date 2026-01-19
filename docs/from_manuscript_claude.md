# What's Missing for Stage 1 Submission

Based on Nature's RR guidelines, Stage 1 requires: Introduction, Methods, and optionally Pilot Data. Your manuscript has the first two; the in-silico results serve as Pilot Data.

Required In-Silico Results

1. Summary statistics from simulations (n=2,200):
- Mean and SD of trial counts for BSA vs Bayesian by phenotype
- Overall efficiency gain (δN) with 95% CI
- Test-retest variability (σ) for each procedure

2. Key figures needed:
- Figure: Efficiency comparison - Box/violin plot showing trials to threshold (BSA vs Bayesian) across 11 phenotypes
- Figure: Reliability comparison - Test-retest difference distributions for both procedures
- Figure: Phenotype-specific predictions - Expected efficiency gain per phenotype (this drives H3)
- Figure: Example posteriors - Showing convergence for different phenotypes (you already have one)

3. Effect size estimates for power analysis justification:
- Currently you cite "δN = 8 trials" and "δσ = 2 dB" and "ρ = 0.6" - these need to be backed by simulation results
- The simulations should show these are realistic expectations

## Suggested Results/Pilot Data Section Structure

\subsection*{Pilot data: In-silico simulation results}

\subsubsection*{Efficiency (H1)}
[Summary table: mean trials per frequency, total trials, by phenotype]
[Figure showing distribution]

\subsubsection*{Reliability (H2)}
[Test-retest statistics from simulations]

\subsubsection*{Phenotype-specific predictions (H3)}
[Table/figure showing predicted efficiency gain per phenotype]

What I Need From You

1. Simulation output data (CSV or summary statistics) showing:
- Trials to threshold for BSA and Bayesian per phenotype
- Test-retest differences
- Any phenotype-specific metrics
2. Figures (PDF preferred) or data to generate them

Once you provide the in-silico results, I can integrate them into the manuscript and create appropriate tables/figure captions.
