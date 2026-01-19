"""
Bayesian hypothesis testing framework for audiometry comparison.

Implements the Bayesian analysis framework specified in the Stage 1 manuscript:
- Bayes Factor calculations for directional hypotheses
- Half-Cauchy priors for effect sizes
- Posterior estimation with HDI
- Probability of practical significance

References:
    Schönbrodt & Wagenmakers (2018). Bayes Factor Design Analysis.
"""

import numpy as np
from scipy import stats
from scipy.special import logsumexp
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, List
import warnings


@dataclass
class BayesFactorResult:
    """Results from Bayes Factor analysis."""
    bf10: float  # Bayes factor for H1 vs H0
    bf01: float  # Bayes factor for H0 vs H1
    log_bf10: float  # Log Bayes factor (more stable)
    interpretation: str  # Verbal interpretation
    posterior_odds: float  # With 1:1 prior odds


@dataclass
class PosteriorResult:
    """Results from posterior estimation."""
    mean: float
    median: float
    sd: float
    hdi_low: float  # 95% HDI lower bound
    hdi_high: float  # 95% HDI upper bound
    samples: np.ndarray  # MCMC samples or grid approximation


@dataclass
class BayesianHypothesisResult:
    """Complete results for a Bayesian hypothesis test."""
    # Bayes Factor
    bf: BayesFactorResult
    # Posterior on effect size
    posterior: PosteriorResult
    # Probability of practical significance
    prob_practical: float
    # Practical significance threshold used
    practical_threshold: float
    # Raw data summary
    effect_observed: float
    effect_se: float
    n: int
    # Frequentist comparison
    frequentist_p: float
    frequentist_d: float  # Cohen's d


def interpret_bf(bf10: float) -> str:
    """
    Interpret Bayes Factor using standard thresholds.

    From Schönbrodt & Wagenmakers (2018).
    """
    if bf10 > 100:
        return "Extreme evidence for H1"
    elif bf10 > 30:
        return "Very strong evidence for H1"
    elif bf10 > 10:
        return "Strong evidence for H1"
    elif bf10 > 3:
        return "Moderate evidence for H1"
    elif bf10 > 1:
        return "Anecdotal evidence for H1"
    elif bf10 > 1/3:
        return "Inconclusive"
    elif bf10 > 1/10:
        return "Moderate evidence for H0"
    elif bf10 > 1/30:
        return "Strong evidence for H0"
    elif bf10 > 1/100:
        return "Very strong evidence for H0"
    else:
        return "Extreme evidence for H0"


def half_cauchy_pdf(x: np.ndarray, scale: float) -> np.ndarray:
    """Half-Cauchy probability density function."""
    return 2 / (np.pi * scale * (1 + (x / scale) ** 2))


def half_cauchy_logpdf(x: np.ndarray, scale: float) -> np.ndarray:
    """Log of Half-Cauchy PDF for numerical stability."""
    return np.log(2) - np.log(np.pi) - np.log(scale) - np.log(1 + (x / scale) ** 2)


def compute_hdi(samples: np.ndarray, credible_mass: float = 0.95) -> Tuple[float, float]:
    """
    Compute Highest Density Interval from samples.

    Parameters
    ----------
    samples : array
        MCMC samples or posterior samples
    credible_mass : float
        Probability mass to include (default 0.95)

    Returns
    -------
    tuple
        (hdi_low, hdi_high)
    """
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    interval_size = int(np.ceil(credible_mass * n))

    # Find the narrowest interval containing credible_mass
    interval_widths = sorted_samples[interval_size:] - sorted_samples[:-interval_size]
    min_idx = np.argmin(interval_widths)

    hdi_low = sorted_samples[min_idx]
    hdi_high = sorted_samples[min_idx + interval_size]

    return hdi_low, hdi_high


def bayes_factor_one_sample_t(
    data: np.ndarray,
    prior_scale: float,
    null_value: float = 0,
    direction: str = "greater"
) -> BayesFactorResult:
    """
    Compute Bayes Factor for one-sample t-test with half-Cauchy prior.

    Uses Savage-Dickey density ratio method.

    Parameters
    ----------
    data : array
        Observed data
    prior_scale : float
        Scale parameter for half-Cauchy prior on effect size
    null_value : float
        Value under null hypothesis (default 0)
    direction : str
        "greater", "less", or "two-sided"

    Returns
    -------
    BayesFactorResult
    """
    n = len(data)
    mean = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)

    # Effect size (standardized)
    effect = (mean - null_value) / se if se > 0 else 0

    # For directional test with half-Cauchy prior
    # Use numerical integration

    # Grid for numerical integration
    n_grid = 10000
    if direction == "greater":
        # Effect > 0
        delta_grid = np.linspace(0.001, 50, n_grid)  # Effect size grid
    elif direction == "less":
        delta_grid = np.linspace(-50, -0.001, n_grid)
    else:
        delta_grid = np.linspace(-50, 50, n_grid)

    # Prior: Half-Cauchy on effect size (in original units)
    prior = half_cauchy_pdf(np.abs(delta_grid) * se, prior_scale)
    prior = prior / np.sum(prior)  # Normalize

    # Likelihood: t-distribution centered at observed effect
    # P(data | delta) ~ t(df, delta, se)
    df = n - 1
    log_likelihood = stats.t.logpdf(effect, df, loc=delta_grid, scale=1)

    # Marginal likelihood under H1
    log_marginal_h1 = logsumexp(log_likelihood + np.log(prior + 1e-300))

    # Marginal likelihood under H0 (delta = 0)
    log_marginal_h0 = stats.t.logpdf(effect, df, loc=0, scale=1)

    # Bayes Factor
    log_bf10 = log_marginal_h1 - log_marginal_h0

    # Adjust for directional hypothesis
    if direction in ["greater", "less"]:
        # One-sided BF is approximately 2x two-sided when effect is in predicted direction
        if (direction == "greater" and mean > null_value) or \
           (direction == "less" and mean < null_value):
            log_bf10 = log_bf10 + np.log(2)
        else:
            log_bf10 = log_bf10 - np.log(2)

    bf10 = np.exp(log_bf10)
    bf01 = 1 / bf10 if bf10 > 0 else np.inf

    return BayesFactorResult(
        bf10=bf10,
        bf01=bf01,
        log_bf10=log_bf10,
        interpretation=interpret_bf(bf10),
        posterior_odds=bf10  # With 1:1 prior odds
    )


def bayes_factor_paired_t(
    x: np.ndarray,
    y: np.ndarray,
    prior_scale: float,
    direction: str = "greater"
) -> BayesFactorResult:
    """
    Compute Bayes Factor for paired t-test (x - y).

    Parameters
    ----------
    x, y : arrays
        Paired observations
    prior_scale : float
        Scale for half-Cauchy prior on mean difference
    direction : str
        "greater" (x > y), "less" (x < y), or "two-sided"

    Returns
    -------
    BayesFactorResult
    """
    diff = np.array(x) - np.array(y)
    return bayes_factor_one_sample_t(diff, prior_scale, null_value=0, direction=direction)


def estimate_posterior(
    data: np.ndarray,
    prior_scale: float,
    n_samples: int = 10000,
    direction: str = "greater"
) -> PosteriorResult:
    """
    Estimate posterior distribution on effect size.

    Uses grid approximation for simplicity.

    Parameters
    ----------
    data : array
        Observed data (e.g., differences)
    prior_scale : float
        Scale for half-Cauchy prior
    n_samples : int
        Number of posterior samples to generate
    direction : str
        "greater", "less", or "two-sided"

    Returns
    -------
    PosteriorResult
    """
    n = len(data)
    mean = np.mean(data)
    se = np.std(data, ddof=1) / np.sqrt(n)
    df = n - 1

    # Grid for posterior
    if direction == "greater":
        grid = np.linspace(0, mean + 5 * se, 5000)
    elif direction == "less":
        grid = np.linspace(mean - 5 * se, 0, 5000)
    else:
        grid = np.linspace(mean - 5 * se, mean + 5 * se, 5000)

    # Prior (half-Cauchy)
    log_prior = half_cauchy_logpdf(np.abs(grid), prior_scale)

    # Likelihood (t-distribution)
    log_likelihood = stats.t.logpdf(mean, df, loc=grid, scale=se)

    # Posterior (unnormalized)
    log_posterior = log_prior + log_likelihood
    log_posterior = log_posterior - logsumexp(log_posterior)  # Normalize

    posterior = np.exp(log_posterior)
    posterior = posterior / np.sum(posterior)

    # Sample from posterior
    samples = np.random.choice(grid, size=n_samples, p=posterior)

    # Compute summary statistics
    hdi_low, hdi_high = compute_hdi(samples)

    return PosteriorResult(
        mean=np.mean(samples),
        median=np.median(samples),
        sd=np.std(samples),
        hdi_low=hdi_low,
        hdi_high=hdi_high,
        samples=samples
    )


def test_h1_efficiency(
    mhw_trials: np.ndarray,
    bayes_trials: np.ndarray,
    prior_scale: float = 10.0,
    practical_threshold: float = 5.0
) -> BayesianHypothesisResult:
    """
    Test H1: Bayesian procedure requires fewer trials.

    δ_N = N_mHW - N_Bayes > 0

    Parameters
    ----------
    mhw_trials : array
        Total trials per listener for mHW
    bayes_trials : array
        Total trials per listener for Bayesian
    prior_scale : float
        Scale for Half-Cauchy prior (default 10 trials)
    practical_threshold : float
        Minimum meaningful difference (default 5 trials)

    Returns
    -------
    BayesianHypothesisResult
    """
    diff = np.array(mhw_trials) - np.array(bayes_trials)
    n = len(diff)

    # Bayes Factor
    bf = bayes_factor_one_sample_t(diff, prior_scale, direction="greater")

    # Posterior
    posterior = estimate_posterior(diff, prior_scale, direction="greater")

    # Probability of practical significance
    prob_practical = np.mean(posterior.samples > practical_threshold)

    # Frequentist comparison
    t_stat, p_val = stats.ttest_1samp(diff, 0)
    p_one_sided = p_val / 2 if np.mean(diff) > 0 else 1 - p_val / 2
    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0

    return BayesianHypothesisResult(
        bf=bf,
        posterior=posterior,
        prob_practical=prob_practical,
        practical_threshold=practical_threshold,
        effect_observed=np.mean(diff),
        effect_se=np.std(diff, ddof=1) / np.sqrt(n),
        n=n,
        frequentist_p=p_one_sided,
        frequentist_d=cohens_d
    )


def test_h2_reliability(
    mhw_sd: float,
    bayes_sd: float,
    n_observations: int,
    prior_scale: float = 3.0,
    practical_threshold: float = 1.0
) -> BayesianHypothesisResult:
    """
    Test H2: Bayesian procedure has better reliability.

    δ_σ = σ_mHW - σ_Bayes > 0

    This uses a different approach since we're comparing SDs.
    We use a Bayesian approach based on the ratio of variances.

    Parameters
    ----------
    mhw_sd : float
        Test-retest SD for mHW
    bayes_sd : float
        Test-retest SD for Bayesian
    n_observations : int
        Number of paired observations (listeners * frequencies)
    prior_scale : float
        Scale for Half-Cauchy prior on SD difference (default 3 dB)
    practical_threshold : float
        Minimum meaningful SD difference (default 1 dB)

    Returns
    -------
    BayesianHypothesisResult
    """
    # Difference in SDs
    diff_sd = mhw_sd - bayes_sd

    # Approximate SE of SD difference using delta method
    # SE(SD) ≈ SD / sqrt(2n)
    se_mhw = mhw_sd / np.sqrt(2 * n_observations)
    se_bayes = bayes_sd / np.sqrt(2 * n_observations)
    se_diff = np.sqrt(se_mhw**2 + se_bayes**2)

    # Create pseudo-data for the difference
    # This is an approximation - ideally would use full bootstrap
    pseudo_samples = np.random.normal(diff_sd, se_diff, 1000)

    # Bayes Factor (using the pseudo-samples)
    bf = bayes_factor_one_sample_t(pseudo_samples, prior_scale, direction="greater")

    # Posterior
    posterior = estimate_posterior(pseudo_samples, prior_scale, direction="greater")

    # Probability of practical significance
    prob_practical = np.mean(posterior.samples > practical_threshold)

    # Frequentist: F-test for variance ratio
    var_ratio = mhw_sd**2 / bayes_sd**2 if bayes_sd > 0 else np.inf
    df = n_observations - 1
    p_val = 1 - stats.f.cdf(var_ratio, df, df)

    return BayesianHypothesisResult(
        bf=bf,
        posterior=posterior,
        prob_practical=prob_practical,
        practical_threshold=practical_threshold,
        effect_observed=diff_sd,
        effect_se=se_diff,
        n=n_observations,
        frequentist_p=p_val,
        frequentist_d=diff_sd / ((mhw_sd + bayes_sd) / 2)  # Standardized difference
    )


def test_h2_reliability_from_differences(
    mhw_differences: np.ndarray,
    bayes_differences: np.ndarray,
    prior_scale: float = 3.0,
    practical_threshold: float = 1.0
) -> BayesianHypothesisResult:
    """
    Test H2 using raw test-retest differences.

    More accurate than using summary statistics.

    Parameters
    ----------
    mhw_differences : array
        Test-retest differences for mHW (all freq * listeners)
    bayes_differences : array
        Test-retest differences for Bayesian
    prior_scale : float
        Scale for Half-Cauchy prior on SD difference
    practical_threshold : float
        Minimum meaningful SD difference

    Returns
    -------
    BayesianHypothesisResult
    """
    # Compute SDs
    mhw_sd = np.std(mhw_differences, ddof=1)
    bayes_sd = np.std(bayes_differences, ddof=1)
    n = len(mhw_differences)

    # Bootstrap the SD difference
    n_bootstrap = 2000
    sd_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        mhw_boot_sd = np.std(mhw_differences[idx], ddof=1)
        bayes_boot_sd = np.std(bayes_differences[idx], ddof=1)
        sd_diffs.append(mhw_boot_sd - bayes_boot_sd)

    sd_diffs = np.array(sd_diffs)

    # Bayes Factor using bootstrap samples
    bf = bayes_factor_one_sample_t(sd_diffs, prior_scale, direction="greater")

    # Posterior from bootstrap
    hdi_low, hdi_high = compute_hdi(sd_diffs)
    posterior = PosteriorResult(
        mean=np.mean(sd_diffs),
        median=np.median(sd_diffs),
        sd=np.std(sd_diffs),
        hdi_low=hdi_low,
        hdi_high=hdi_high,
        samples=sd_diffs
    )

    # Probability of practical significance
    prob_practical = np.mean(sd_diffs > practical_threshold)

    # Frequentist
    var_ratio = mhw_sd**2 / bayes_sd**2 if bayes_sd > 0 else np.inf
    df = n - 1
    p_val = 1 - stats.f.cdf(var_ratio, df, df)

    return BayesianHypothesisResult(
        bf=bf,
        posterior=posterior,
        prob_practical=prob_practical,
        practical_threshold=practical_threshold,
        effect_observed=mhw_sd - bayes_sd,
        effect_se=np.std(sd_diffs),
        n=n,
        frequentist_p=p_val,
        frequentist_d=(mhw_sd - bayes_sd) / ((mhw_sd + bayes_sd) / 2)
    )


def bayesian_icc(
    test1: np.ndarray,
    test2: np.ndarray,
    n_samples: int = 5000
) -> Dict:
    """
    Estimate ICC with Bayesian credible interval.

    Uses variance components estimation.

    Parameters
    ----------
    test1, test2 : arrays
        Test and retest measurements (can be flattened)
    n_samples : int
        Number of bootstrap samples for CI

    Returns
    -------
    dict
        icc, hdi_low, hdi_high, prob_excellent (>0.9), prob_good (>0.75)
    """
    test1 = np.asarray(test1).flatten()
    test2 = np.asarray(test2).flatten()
    n = len(test1)

    # Bootstrap ICC
    icc_samples = []
    for _ in range(n_samples):
        idx = np.random.choice(n, n, replace=True)
        t1 = test1[idx]
        t2 = test2[idx]

        # Simple ICC(2,1) calculation
        grand_mean = (np.mean(t1) + np.mean(t2)) / 2
        subject_means = (t1 + t2) / 2

        # Variance components
        var_between = np.var(subject_means, ddof=1)
        var_within = np.mean((t1 - subject_means)**2 + (t2 - subject_means)**2) / 2

        # ICC(2,1)
        icc = var_between / (var_between + var_within) if (var_between + var_within) > 0 else 0
        icc_samples.append(icc)

    icc_samples = np.array(icc_samples)
    hdi_low, hdi_high = compute_hdi(icc_samples)

    return {
        'icc': np.median(icc_samples),
        'icc_mean': np.mean(icc_samples),
        'hdi_low': hdi_low,
        'hdi_high': hdi_high,
        'prob_excellent': np.mean(icc_samples > 0.9),
        'prob_good': np.mean(icc_samples > 0.75),
        'samples': icc_samples
    }


def format_bayesian_results(result: BayesianHypothesisResult, hypothesis_name: str) -> str:
    """Format Bayesian hypothesis test results for reporting."""
    lines = [
        f"\n{hypothesis_name}",
        "=" * len(hypothesis_name),
        "",
        "Bayesian Analysis:",
        f"  BF10 = {result.bf.bf10:.2f} ({result.bf.interpretation})",
        f"  Posterior mean = {result.posterior.mean:.2f}",
        f"  95% HDI = [{result.posterior.hdi_low:.2f}, {result.posterior.hdi_high:.2f}]",
        f"  P(effect > {result.practical_threshold}) = {result.prob_practical:.1%}",
        "",
        "Frequentist Comparison:",
        f"  Observed effect = {result.effect_observed:.2f} (SE = {result.effect_se:.2f})",
        f"  Cohen's d = {result.frequentist_d:.2f}",
        f"  p-value (one-sided) = {result.frequentist_p:.2e}",
        f"  n = {result.n}",
    ]
    return "\n".join(lines)


# Convenience function for full analysis
def run_bayesian_analysis(
    mhw_trials: np.ndarray,
    bayes_trials: np.ndarray,
    mhw_test_retest_diff: np.ndarray,
    bayes_test_retest_diff: np.ndarray,
    mhw_test1: np.ndarray = None,
    mhw_test2: np.ndarray = None,
    bayes_test1: np.ndarray = None,
    bayes_test2: np.ndarray = None,
) -> Dict:
    """
    Run complete Bayesian analysis for H1 and H2.

    Parameters
    ----------
    mhw_trials, bayes_trials : arrays
        Total trials per listener
    mhw_test_retest_diff, bayes_test_retest_diff : arrays
        Test-retest threshold differences (all freqs flattened)
    mhw_test1, mhw_test2, bayes_test1, bayes_test2 : arrays, optional
        Raw threshold data for ICC calculation

    Returns
    -------
    dict
        Complete analysis results
    """
    results = {}

    # H1: Efficiency
    h1 = test_h1_efficiency(
        mhw_trials=mhw_trials,
        bayes_trials=bayes_trials,
        prior_scale=10.0,  # Half-Cauchy(0, 10) as per manuscript
        practical_threshold=5.0
    )
    results['h1'] = h1
    results['h1_summary'] = format_bayesian_results(h1, "H1: Efficiency (δN = N_mHW - N_Bayes)")

    # H2: Reliability
    h2 = test_h2_reliability_from_differences(
        mhw_differences=mhw_test_retest_diff,
        bayes_differences=bayes_test_retest_diff,
        prior_scale=3.0,  # Half-Cauchy(0, 3) as per manuscript
        practical_threshold=1.0
    )
    results['h2'] = h2
    results['h2_summary'] = format_bayesian_results(h2, "H2: Reliability (δσ = σ_mHW - σ_Bayes)")

    # Bayesian ICC estimates
    if mhw_test1 is not None and mhw_test2 is not None:
        results['icc_mhw'] = bayesian_icc(mhw_test1, mhw_test2)
    if bayes_test1 is not None and bayes_test2 is not None:
        results['icc_bayes'] = bayesian_icc(bayes_test1, bayes_test2)

    return results
