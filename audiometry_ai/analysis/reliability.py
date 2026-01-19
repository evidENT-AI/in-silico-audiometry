"""
Reliability analysis for audiometric procedures.

Implements:
- Intraclass Correlation Coefficient (ICC)
- Bland-Altman analysis
- Test-retest statistics

These are used for H2 hypothesis testing (reliability comparison).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats


@dataclass
class ICCResult:
    """Results from ICC calculation."""
    icc: float  # ICC value
    icc_type: str  # ICC type (e.g., "ICC(2,1)")
    ci_low: float  # 95% CI lower bound
    ci_high: float  # 95% CI upper bound
    f_value: float  # F statistic
    p_value: float  # p-value for F test
    n_subjects: int
    n_raters: int  # Or measurements (for test-retest, n_raters=2)


@dataclass
class BlandAltmanResult:
    """Results from Bland-Altman analysis."""
    mean_diff: float  # Mean difference (bias)
    std_diff: float  # SD of differences
    loa_lower: float  # Lower limit of agreement (mean - 1.96*SD)
    loa_upper: float  # Upper limit of agreement (mean + 1.96*SD)
    ci_mean_diff: Tuple[float, float]  # 95% CI for mean difference
    ci_loa_lower: Tuple[float, float]  # 95% CI for lower LOA
    ci_loa_upper: Tuple[float, float]  # 95% CI for upper LOA
    n: int


def compute_icc(
    test1: np.ndarray,
    test2: np.ndarray,
    icc_type: str = "ICC(2,1)"
) -> ICCResult:
    """
    Compute Intraclass Correlation Coefficient for test-retest reliability.

    Parameters
    ----------
    test1 : np.ndarray
        First measurement for each subject
    test2 : np.ndarray
        Second measurement for each subject
    icc_type : str
        ICC type: "ICC(1,1)", "ICC(2,1)", or "ICC(3,1)"
        - ICC(1,1): One-way random effects, absolute agreement
        - ICC(2,1): Two-way random effects, absolute agreement (recommended)
        - ICC(3,1): Two-way mixed effects, consistency

    Returns
    -------
    ICCResult
        ICC value with confidence intervals and statistics

    Notes
    -----
    For test-retest reliability with 2 measurements:
    - ICC(2,1) is typically recommended
    - Interpretation: ICC < 0.5 poor, 0.5-0.75 moderate,
      0.75-0.9 good, > 0.9 excellent (Koo & Li, 2016)
    """
    test1 = np.asarray(test1).flatten()
    test2 = np.asarray(test2).flatten()

    if len(test1) != len(test2):
        raise ValueError("test1 and test2 must have same length")

    n = len(test1)
    k = 2  # Number of measurements (test-retest)

    # Combine into matrix: rows = subjects, cols = measurements
    data = np.column_stack([test1, test2])

    # Compute ANOVA components
    # Grand mean
    grand_mean = np.mean(data)

    # Subject means
    subject_means = np.mean(data, axis=1)

    # Rater/measurement means
    rater_means = np.mean(data, axis=0)

    # Sum of squares
    # Between-subjects (rows)
    ss_between = k * np.sum((subject_means - grand_mean) ** 2)

    # Within-subjects
    ss_within = np.sum((data - subject_means[:, np.newaxis]) ** 2)

    # Between-raters (columns)
    ss_raters = n * np.sum((rater_means - grand_mean) ** 2)

    # Error (residual)
    ss_error = ss_within - ss_raters

    # Degrees of freedom
    df_between = n - 1
    df_within = n * (k - 1)
    df_raters = k - 1
    df_error = (n - 1) * (k - 1)

    # Mean squares
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    ms_raters = ss_raters / df_raters if df_raters > 0 else 0
    ms_error = ss_error / df_error if df_error > 0 else 0

    # Compute ICC based on type
    if icc_type == "ICC(1,1)":
        # One-way random effects, absolute agreement, single measurement
        icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
        f_value = ms_between / ms_within
        df1, df2 = df_between, df_within

    elif icc_type == "ICC(2,1)":
        # Two-way random effects, absolute agreement, single measurement
        icc = (ms_between - ms_error) / (
            ms_between + (k - 1) * ms_error + (k / n) * (ms_raters - ms_error)
        )
        # Avoid division issues
        if ms_error > 0:
            f_value = ms_between / ms_error
        else:
            f_value = np.inf
        df1, df2 = df_between, df_error

    elif icc_type == "ICC(3,1)":
        # Two-way mixed effects, consistency, single measurement
        icc = (ms_between - ms_error) / (ms_between + (k - 1) * ms_error)
        if ms_error > 0:
            f_value = ms_between / ms_error
        else:
            f_value = np.inf
        df1, df2 = df_between, df_error

    else:
        raise ValueError(f"Unknown ICC type: {icc_type}")

    # Confidence interval using F distribution
    # Based on Shrout & Fleiss (1979)
    alpha = 0.05

    if f_value > 0 and np.isfinite(f_value):
        f_low = f_value / stats.f.ppf(1 - alpha / 2, df1, df2)
        f_high = f_value * stats.f.ppf(1 - alpha / 2, df2, df1)

        # Convert F to ICC
        if icc_type == "ICC(1,1)":
            ci_low = (f_low - 1) / (f_low + k - 1)
            ci_high = (f_high - 1) / (f_high + k - 1)
        else:
            # Approximation for ICC(2,1) and ICC(3,1)
            ci_low = (f_low - 1) / (f_low + k - 1)
            ci_high = (f_high - 1) / (f_high + k - 1)

        # p-value
        p_value = 1 - stats.f.cdf(f_value, df1, df2)
    else:
        ci_low, ci_high = -1.0, 1.0
        p_value = 1.0

    # Ensure ICC and CIs are in valid range
    icc = np.clip(icc, -1.0, 1.0)
    ci_low = np.clip(ci_low, -1.0, 1.0)
    ci_high = np.clip(ci_high, -1.0, 1.0)

    return ICCResult(
        icc=float(icc),
        icc_type=icc_type,
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        f_value=float(f_value),
        p_value=float(p_value),
        n_subjects=n,
        n_raters=k
    )


def bland_altman_stats(
    test1: np.ndarray,
    test2: np.ndarray
) -> BlandAltmanResult:
    """
    Compute Bland-Altman statistics for method agreement.

    Parameters
    ----------
    test1 : np.ndarray
        First measurement for each subject
    test2 : np.ndarray
        Second measurement for each subject

    Returns
    -------
    BlandAltmanResult
        Bland-Altman statistics including limits of agreement

    Notes
    -----
    For audiometry, LOA within ±5 dB is typically considered acceptable
    clinical agreement (ISO 8253-1).
    """
    test1 = np.asarray(test1).flatten()
    test2 = np.asarray(test2).flatten()

    if len(test1) != len(test2):
        raise ValueError("test1 and test2 must have same length")

    n = len(test1)
    diff = test1 - test2
    mean_val = (test1 + test2) / 2  # Not used in basic stats but for plots

    # Mean and SD of differences
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    # Limits of agreement
    loa_lower = mean_diff - 1.96 * std_diff
    loa_upper = mean_diff + 1.96 * std_diff

    # Standard errors for CIs
    se_mean = std_diff / np.sqrt(n)
    se_loa = np.sqrt(3 * std_diff ** 2 / n)  # Approximate

    # 95% CIs
    t_crit = stats.t.ppf(0.975, n - 1)

    ci_mean_diff = (
        mean_diff - t_crit * se_mean,
        mean_diff + t_crit * se_mean
    )

    ci_loa_lower = (
        loa_lower - t_crit * se_loa,
        loa_lower + t_crit * se_loa
    )

    ci_loa_upper = (
        loa_upper - t_crit * se_loa,
        loa_upper + t_crit * se_loa
    )

    return BlandAltmanResult(
        mean_diff=float(mean_diff),
        std_diff=float(std_diff),
        loa_lower=float(loa_lower),
        loa_upper=float(loa_upper),
        ci_mean_diff=ci_mean_diff,
        ci_loa_lower=ci_loa_lower,
        ci_loa_upper=ci_loa_upper,
        n=n
    )


def compute_test_retest_reliability(
    results: List[Dict],
    procedure: str = 'bayesian',
    frequencies: Optional[List[int]] = None
) -> Dict:
    """
    Compute comprehensive test-retest reliability metrics.

    Parameters
    ----------
    results : list of dict
        Results from simulation with test-retest data
    procedure : str
        'bayesian' or 'mhw'
    frequencies : list of int, optional
        Frequencies to analyze (default: all available)

    Returns
    -------
    dict
        Comprehensive reliability metrics including ICC and Bland-Altman
    """
    # Extract test-retest pairs
    test1_all = []
    test2_all = []

    # Group by listener
    by_listener = {}
    for r in results:
        lid = r.get('listener_id')
        if lid not in by_listener:
            by_listener[lid] = []
        by_listener[lid].append(r)

    # Get test-retest pairs
    for lid, sessions in by_listener.items():
        if len(sessions) < 2:
            continue

        # Get thresholds for each session
        s1 = sessions[0]
        s2 = sessions[1]

        if procedure == 'bayesian':
            t1 = s1.get('bayes_thresholds', {})
            t2 = s2.get('bayes_thresholds', {})
        else:
            t1 = s1.get('mhw_thresholds', {})
            t2 = s2.get('mhw_thresholds', {})

        # Average across frequencies or collect all
        if frequencies is None:
            frequencies = list(set(t1.keys()) & set(t2.keys()))

        for freq in frequencies:
            if freq in t1 and freq in t2:
                test1_all.append(t1[freq])
                test2_all.append(t2[freq])

    test1_all = np.array(test1_all)
    test2_all = np.array(test2_all)

    if len(test1_all) < 3:
        return {
            'error': 'Insufficient data for reliability analysis',
            'n_pairs': len(test1_all)
        }

    # Compute ICC
    icc_result = compute_icc(test1_all, test2_all, "ICC(2,1)")

    # Compute Bland-Altman
    ba_result = bland_altman_stats(test1_all, test2_all)

    # Test-retest SD (used in H2)
    test_retest_sd = ba_result.std_diff

    return {
        'icc': icc_result.icc,
        'icc_ci': (icc_result.ci_low, icc_result.ci_high),
        'icc_type': icc_result.icc_type,
        'test_retest_sd': test_retest_sd,
        'mean_diff': ba_result.mean_diff,
        'loa': (ba_result.loa_lower, ba_result.loa_upper),
        'n_pairs': len(test1_all),
        'icc_result': icc_result,
        'bland_altman_result': ba_result
    }


def interpret_icc(icc: float) -> str:
    """
    Interpret ICC value according to Koo & Li (2016) guidelines.

    Returns interpretation string.
    """
    if icc < 0.5:
        return "poor"
    elif icc < 0.75:
        return "moderate"
    elif icc < 0.9:
        return "good"
    else:
        return "excellent"
