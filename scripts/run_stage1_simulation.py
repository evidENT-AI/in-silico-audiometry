#!/usr/bin/env python
"""
Stage 1 Manuscript Simulation Runner

This script runs the complete Stage 1 simulation for the Registered Report:
1. Generates 2200 virtual listeners across 9 GMM-derived phenotypes
2. Runs both mHW and Bayesian procedures (test-retest)
3. Computes H1 (efficiency), H2 (reliability), H3 (phenotype matching) statistics
4. Generates manuscript figures

Usage:
    python scripts/run_stage1_simulation.py [--n_listeners N] [--seed SEED] [--output_dir DIR]
    python scripts/run_stage1_simulation.py --mini  # Quick test with 50 listeners
"""

import argparse
import json
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import warnings
from tqdm import tqdm

# Local imports
from audiometry_ai.simulation.phenotypes import (
    PhenotypeGenerator,
    PsychometricParameterGenerator,
    PHENOTYPE_DEFINITIONS,
    get_phenotype_names,
    FREQUENCIES,
)
from audiometry_ai.simulation.response_model import HearingResponseModel
from audiometry_ai.procedures.basic_bayes import BayesianPureToneAudiometry
from audiometry_ai.analysis.reliability import (
    compute_icc,
    bland_altman_stats,
    ICCResult,
    BlandAltmanResult,
)
from audiometry_ai.analysis.phenotype_matching import (
    PhenotypeMatching,
    cross_validate_matching,
)
from audiometry_ai.analysis.bayesian_hypothesis import (
    test_h1_efficiency,
    test_h2_reliability_from_differences,
    bayesian_icc,
    format_bayesian_results,
)


@dataclass
class ListenerResult:
    """Results for a single listener."""
    listener_id: int
    phenotype: str
    category: str
    true_thresholds: Dict[int, float]

    # Bayesian results
    bayes_thresholds: Dict[int, float]
    bayes_uncertainties: Dict[int, float]
    bayes_trials_per_freq: Dict[int, int]
    bayes_total_trials: int

    # mHW results (simulated based on typical mHW behavior)
    mhw_thresholds: Dict[int, float]
    mhw_trials_per_freq: Dict[int, int]
    mhw_total_trials: int

    # Errors
    bayes_errors: Dict[int, float]
    mhw_errors: Dict[int, float]
    bayes_mean_error: float
    mhw_mean_error: float

    # Psychometric parameters
    slope: float
    false_positive_rate: float
    false_negative_rate: float


def simulate_mhw_procedure(
    true_thresholds: Dict[int, float],
    response_model: HearingResponseModel,
    frequencies: List[int],
    rng: np.random.Generator
) -> Tuple[Dict[int, float], Dict[int, int]]:
    """
    Simulate modified Hughson-Westlake procedure.

    This is a simplified simulation that captures the key characteristics
    of the mHW procedure without full implementation.
    """
    thresholds = {}
    trials_per_freq = {}

    for freq in frequencies:
        true_thresh = true_thresholds[freq]

        # mHW typically takes 8-15 trials per frequency
        # More trials needed for higher thresholds and more variable responses
        base_trials = 10
        threshold_factor = 1 + abs(true_thresh - 40) / 100  # More trials for extreme thresholds
        n_trials = int(base_trials * threshold_factor + rng.integers(-2, 3))
        n_trials = max(6, min(20, n_trials))

        # Simulate threshold estimation with typical mHW accuracy
        # mHW has ~3-5 dB SD error
        error_sd = 4.0
        estimated_thresh = true_thresh + rng.normal(0, error_sd)
        estimated_thresh = round(estimated_thresh / 5) * 5  # Round to 5 dB
        estimated_thresh = np.clip(estimated_thresh, -10, 120)

        thresholds[freq] = float(estimated_thresh)
        trials_per_freq[freq] = n_trials

    return thresholds, trials_per_freq


def run_listener_simulation(
    listener_id: int,
    audiogram: Dict[int, float],
    phenotype: str,
    category: str,
    psych_params: Dict[str, float],
    frequencies: List[int],
    seed: int,
    use_nhanes_priors: bool = False
) -> ListenerResult:
    """Run simulation for a single listener."""
    rng = np.random.default_rng(seed)

    # Create response model with listener's psychometric parameters
    # Note: HearingResponseModel uses guess_rate (FP) and lapse_rate (FN)
    response_model = HearingResponseModel(
        slope=psych_params['slope'],
        guess_rate=psych_params['false_positive_rate'],
        lapse_rate=psych_params['false_negative_rate']
    )

    # Run Bayesian procedure
    bayes_procedure = BayesianPureToneAudiometry(
        hearing_profile_data=audiogram,
        response_model_params={
            'slope': psych_params['slope'],
            'guess_rate': psych_params['false_positive_rate'],
            'lapse_rate': psych_params['false_negative_rate'],
        },
        test_frequencies=frequencies,
        convergence_threshold_db=5.0,
        max_trials_per_freq=30,
        random_state=seed,
        use_nhanes_priors=use_nhanes_priors,
    )

    bayes_results = bayes_procedure.perform_test()

    # Extract Bayesian results
    bayes_thresholds = bayes_results['thresholds']
    bayes_uncertainties = bayes_results['uncertainties']
    bayes_trials_per_freq = {
        freq: len(prog)
        for freq, prog in bayes_results['progression_patterns'].items()
    }
    bayes_total_trials = sum(bayes_trials_per_freq.values())

    # Simulate mHW procedure
    mhw_thresholds, mhw_trials_per_freq = simulate_mhw_procedure(
        audiogram, response_model, frequencies, rng
    )
    mhw_total_trials = sum(mhw_trials_per_freq.values())

    # Calculate errors
    bayes_errors = {
        freq: abs(bayes_thresholds[freq] - audiogram[freq])
        for freq in frequencies
    }
    mhw_errors = {
        freq: abs(mhw_thresholds[freq] - audiogram[freq])
        for freq in frequencies
    }

    return ListenerResult(
        listener_id=listener_id,
        phenotype=phenotype,
        category=category,
        true_thresholds=audiogram,
        bayes_thresholds=bayes_thresholds,
        bayes_uncertainties=bayes_uncertainties,
        bayes_trials_per_freq=bayes_trials_per_freq,
        bayes_total_trials=bayes_total_trials,
        mhw_thresholds=mhw_thresholds,
        mhw_trials_per_freq=mhw_trials_per_freq,
        mhw_total_trials=mhw_total_trials,
        bayes_errors=bayes_errors,
        mhw_errors=mhw_errors,
        bayes_mean_error=np.mean(list(bayes_errors.values())),
        mhw_mean_error=np.mean(list(mhw_errors.values())),
        slope=psych_params['slope'],
        false_positive_rate=psych_params['false_positive_rate'],
        false_negative_rate=psych_params['false_negative_rate'],
    )


def compute_h1_statistics(
    results: List[ListenerResult]
) -> Dict:
    """Compute H1 (efficiency) statistics."""
    bayes_trials = [r.bayes_total_trials for r in results]
    mhw_trials = [r.mhw_total_trials for r in results]

    bayes_errors = [r.bayes_mean_error for r in results]
    mhw_errors = [r.mhw_mean_error for r in results]

    # Trial count comparison
    trial_diff = np.array(mhw_trials) - np.array(bayes_trials)
    mean_reduction = np.mean(trial_diff)
    reduction_pct = mean_reduction / np.mean(mhw_trials) * 100

    # Paired t-test for trials
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(mhw_trials, bayes_trials)

    # Cohen's d effect size
    pooled_std = np.sqrt((np.var(mhw_trials) + np.var(bayes_trials)) / 2)
    cohens_d = mean_reduction / pooled_std if pooled_std > 0 else 0

    # By phenotype
    by_phenotype = {}
    for phenotype in get_phenotype_names():
        phenotype_results = [r for r in results if r.phenotype == phenotype]
        if phenotype_results:
            p_bayes = np.mean([r.bayes_total_trials for r in phenotype_results])
            p_mhw = np.mean([r.mhw_total_trials for r in phenotype_results])
            by_phenotype[phenotype] = {
                'bayes_trials': p_bayes,
                'mhw_trials': p_mhw,
                'reduction': p_mhw - p_bayes,
                'reduction_pct': (p_mhw - p_bayes) / p_mhw * 100 if p_mhw > 0 else 0,
            }

    return {
        'bayes_mean_trials': np.mean(bayes_trials),
        'bayes_std_trials': np.std(bayes_trials),
        'mhw_mean_trials': np.mean(mhw_trials),
        'mhw_std_trials': np.std(mhw_trials),
        'mean_reduction': mean_reduction,
        'reduction_pct': reduction_pct,
        't_statistic': t_stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'bayes_mean_error': np.mean(bayes_errors),
        'mhw_mean_error': np.mean(mhw_errors),
        'by_phenotype': by_phenotype,
    }


def compute_h2_statistics(
    test1_results: List[ListenerResult],
    test2_results: List[ListenerResult]
) -> Dict:
    """Compute H2 (reliability) statistics."""
    # Collect test-retest pairs for Bayesian
    bayes_test1 = []
    bayes_test2 = []
    mhw_test1 = []
    mhw_test2 = []

    for r1, r2 in zip(test1_results, test2_results):
        for freq in FREQUENCIES:
            bayes_test1.append(r1.bayes_thresholds[freq])
            bayes_test2.append(r2.bayes_thresholds[freq])
            mhw_test1.append(r1.mhw_thresholds[freq])
            mhw_test2.append(r2.mhw_thresholds[freq])

    # ICC for Bayesian
    icc_bayes = compute_icc(np.array(bayes_test1), np.array(bayes_test2))
    ba_bayes = bland_altman_stats(np.array(bayes_test1), np.array(bayes_test2))

    # ICC for mHW
    icc_mhw = compute_icc(np.array(mhw_test1), np.array(mhw_test2))
    ba_mhw = bland_altman_stats(np.array(mhw_test1), np.array(mhw_test2))

    return {
        'icc_bayes': icc_bayes.icc,
        'icc_bayes_ci': (icc_bayes.ci_low, icc_bayes.ci_high),
        'icc_mhw': icc_mhw.icc,
        'icc_mhw_ci': (icc_mhw.ci_low, icc_mhw.ci_high),
        'bayes_test_retest_sd': ba_bayes.std_diff,
        'mhw_test_retest_sd': ba_mhw.std_diff,
        'bayes_bias': ba_bayes.mean_diff,
        'mhw_bias': ba_mhw.mean_diff,
        'bayes_loa': (ba_bayes.loa_lower, ba_bayes.loa_upper),
        'mhw_loa': (ba_mhw.loa_lower, ba_mhw.loa_upper),
        'icc_bayes_result': asdict(icc_bayes) if hasattr(icc_bayes, '__dict__') else None,
        'icc_mhw_result': asdict(icc_mhw) if hasattr(icc_mhw, '__dict__') else None,
        'ba_bayes_result': asdict(ba_bayes) if hasattr(ba_bayes, '__dict__') else None,
        'ba_mhw_result': asdict(ba_mhw) if hasattr(ba_mhw, '__dict__') else None,
        # Raw data for Bayesian analysis
        'bayes_test1': bayes_test1,
        'bayes_test2': bayes_test2,
        'mhw_test1': mhw_test1,
        'mhw_test2': mhw_test2,
    }


def compute_bayesian_statistics(
    test1_results: List[ListenerResult],
    test2_results: List[ListenerResult],
    h2_stats: Dict
) -> Dict:
    """
    Compute Bayesian hypothesis testing statistics.

    Implements the Bayesian analysis framework from the manuscript:
    - H1: Half-Cauchy(0, 10) prior on efficiency gain
    - H2: Half-Cauchy(0, 3) prior on reliability improvement

    Returns Bayes Factors, posterior distributions, and probabilities of practical significance.
    """
    # Get trial counts for H1
    mhw_trials = np.array([r.mhw_total_trials for r in test1_results])
    bayes_trials = np.array([r.bayes_total_trials for r in test1_results])

    # Get test-retest differences for H2
    mhw_diff = np.array(h2_stats['mhw_test1']) - np.array(h2_stats['mhw_test2'])
    bayes_diff = np.array(h2_stats['bayes_test1']) - np.array(h2_stats['bayes_test2'])

    # H1: Efficiency - Bayesian analysis
    h1_bayes = test_h1_efficiency(
        mhw_trials=mhw_trials,
        bayes_trials=bayes_trials,
        prior_scale=10.0,  # Half-Cauchy(0, 10) as per manuscript
        practical_threshold=5.0  # 5 trials is practically meaningful
    )

    # H2: Reliability - Bayesian analysis
    h2_bayes = test_h2_reliability_from_differences(
        mhw_differences=mhw_diff,
        bayes_differences=bayes_diff,
        prior_scale=3.0,  # Half-Cauchy(0, 3) as per manuscript
        practical_threshold=1.0  # 1 dB improvement is practically meaningful
    )

    # Bayesian ICC estimates
    icc_bayes_bayesian = bayesian_icc(
        np.array(h2_stats['bayes_test1']),
        np.array(h2_stats['bayes_test2'])
    )
    icc_mhw_bayesian = bayesian_icc(
        np.array(h2_stats['mhw_test1']),
        np.array(h2_stats['mhw_test2'])
    )

    return {
        # H1 Bayesian results
        'h1_bf10': h1_bayes.bf.bf10,
        'h1_bf_interpretation': h1_bayes.bf.interpretation,
        'h1_posterior_mean': h1_bayes.posterior.mean,
        'h1_posterior_hdi': (h1_bayes.posterior.hdi_low, h1_bayes.posterior.hdi_high),
        'h1_prob_practical': h1_bayes.prob_practical,
        'h1_practical_threshold': h1_bayes.practical_threshold,

        # H2 Bayesian results
        'h2_bf10': h2_bayes.bf.bf10,
        'h2_bf_interpretation': h2_bayes.bf.interpretation,
        'h2_posterior_mean': h2_bayes.posterior.mean,
        'h2_posterior_hdi': (h2_bayes.posterior.hdi_low, h2_bayes.posterior.hdi_high),
        'h2_prob_practical': h2_bayes.prob_practical,
        'h2_practical_threshold': h2_bayes.practical_threshold,

        # Bayesian ICC
        'icc_bayes_bayesian': icc_bayes_bayesian['icc'],
        'icc_bayes_bayesian_hdi': (icc_bayes_bayesian['hdi_low'], icc_bayes_bayesian['hdi_high']),
        'icc_bayes_prob_excellent': icc_bayes_bayesian['prob_excellent'],
        'icc_mhw_bayesian': icc_mhw_bayesian['icc'],
        'icc_mhw_bayesian_hdi': (icc_mhw_bayesian['hdi_low'], icc_mhw_bayesian['hdi_high']),
        'icc_mhw_prob_excellent': icc_mhw_bayesian['prob_excellent'],

        # Full results for detailed analysis
        'h1_result': h1_bayes,
        'h2_result': h2_bayes,
    }


def compute_h3_statistics(results: List[ListenerResult]) -> Dict:
    """Compute H3 (phenotype matching) statistics."""
    # Prepare data for matching
    simulation_data = []
    for r in results:
        simulation_data.append({
            'phenotype': r.phenotype,
            'true_thresholds': r.true_thresholds,
            'audiogram': r.true_thresholds,
            'mhw_results': {'trial_counts': r.mhw_trials_per_freq},
            'bayes_results': {'progression_patterns': {
                f: list(range(r.bayes_trials_per_freq.get(f, 10)))
                for f in r.bayes_trials_per_freq
            }},
            'efficiency_gain': r.mhw_total_trials - r.bayes_total_trials,
        })

    # Cross-validation of phenotype matching
    cv_results = cross_validate_matching(
        simulation_data,
        phenotype_key='phenotype',
        n_folds=5,
        random_state=42
    )

    # Fit matcher and compute H3 correlation
    matcher = PhenotypeMatching()
    matcher.fit_centroids(simulation_data, phenotype_key='phenotype')

    # Get predicted vs observed efficiency gains
    predicted_gains = []
    observed_gains = []
    phenotypes = []

    for r in results:
        features = matcher.extract_features(
            audiogram=r.true_thresholds,
            mhw_results={'trial_counts': r.mhw_trials_per_freq},
            bayes_results={'progression_patterns': {
                f: list(range(r.bayes_trials_per_freq.get(f, 10)))
                for f in r.bayes_trials_per_freq
            }}
        )
        matched_phenotype, _ = matcher.match(features)
        pred_gain = matcher.get_predicted_efficiency_gain(matched_phenotype)
        obs_gain = r.mhw_total_trials - r.bayes_total_trials

        predicted_gains.append(pred_gain)
        observed_gains.append(obs_gain)
        phenotypes.append(r.phenotype)

    # Compute correlation
    from scipy import stats
    r_value, p_value = stats.pearsonr(predicted_gains, observed_gains)

    # 95% CI using Fisher z-transform
    n = len(predicted_gains)
    z = np.arctanh(r_value)
    se = 1 / np.sqrt(n - 3) if n > 3 else 1
    z_low, z_high = z - 1.96 * se, z + 1.96 * se
    ci_low, ci_high = np.tanh(z_low), np.tanh(z_high)

    return {
        'matching_accuracy': cv_results['accuracy'],
        'cv_n_correct': cv_results['n_correct'],
        'cv_n_total': cv_results['n_total'],
        'correlation': r_value,
        'correlation_p_value': p_value,
        'correlation_ci': (ci_low, ci_high),
        'predicted_gains': predicted_gains,
        'observed_gains': observed_gains,
        'phenotypes': phenotypes,
    }


def run_simulation(
    n_listeners: int = 2200,
    seed: int = 42,
    output_dir: Path = None,
    use_nhanes_priors: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Run the complete Stage 1 simulation.

    Parameters
    ----------
    n_listeners : int
        Total number of listeners (default 2200)
    seed : int
        Random seed
    output_dir : Path
        Directory to save results
    use_nhanes_priors : bool
        Whether to use NHANES priors for Bayesian procedure
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Complete simulation results
    """
    rng = np.random.default_rng(seed)

    if output_dir is None:
        output_dir = Path("results/stage1_simulation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize generators
    phenotype_gen = PhenotypeGenerator()
    psych_gen = PsychometricParameterGenerator()

    # Calculate listeners per phenotype (proportional to n_target)
    total_target = sum(p.n_target for p in PHENOTYPE_DEFINITIONS.values())
    n_per_phenotype = {
        name: max(1, int(n_listeners * defn.n_target / total_target))
        for name, defn in PHENOTYPE_DEFINITIONS.items()
    }

    # Adjust to match exact total
    current_total = sum(n_per_phenotype.values())
    diff = n_listeners - current_total
    if diff != 0:
        # Add/remove from largest phenotype
        largest = max(n_per_phenotype, key=n_per_phenotype.get)
        n_per_phenotype[largest] += diff

    if verbose:
        print(f"Stage 1 Simulation")
        print(f"==================")
        print(f"Total listeners: {n_listeners}")
        print(f"Random seed: {seed}")
        print(f"NHANES priors: {use_nhanes_priors}")
        print(f"\nPhenotype distribution:")
        for name, n in n_per_phenotype.items():
            print(f"  {name}: {n}")
        print()

    # Generate population
    if verbose:
        print("Generating population...")
    population = phenotype_gen.generate_population(n_per_phenotype, seed=seed)

    # Generate psychometric parameters
    psych_params_list = [psych_gen.generate(rng) for _ in population]

    # Run test session 1
    if verbose:
        print("\nRunning test session 1...")
    test1_results = []
    for i, (listener, psych_params) in enumerate(tqdm(
        zip(population, psych_params_list),
        total=len(population),
        disable=not verbose
    )):
        result = run_listener_simulation(
            listener_id=listener['listener_id'],
            audiogram=listener['audiogram'],
            phenotype=listener['phenotype'],
            category=listener['category'],
            psych_params=psych_params,
            frequencies=FREQUENCIES,
            seed=seed + i,
            use_nhanes_priors=use_nhanes_priors,
        )
        test1_results.append(result)

    # Run test session 2 (test-retest)
    if verbose:
        print("\nRunning test session 2 (retest)...")
    test2_results = []
    for i, (listener, psych_params) in enumerate(tqdm(
        zip(population, psych_params_list),
        total=len(population),
        disable=not verbose
    )):
        result = run_listener_simulation(
            listener_id=listener['listener_id'],
            audiogram=listener['audiogram'],
            phenotype=listener['phenotype'],
            category=listener['category'],
            psych_params=psych_params,
            frequencies=FREQUENCIES,
            seed=seed + len(population) + i,  # Different seed for retest
            use_nhanes_priors=use_nhanes_priors,
        )
        test2_results.append(result)

    # Compute statistics
    if verbose:
        print("\nComputing H1 (efficiency) statistics...")
    h1_stats = compute_h1_statistics(test1_results)

    if verbose:
        print("Computing H2 (reliability) statistics...")
    h2_stats = compute_h2_statistics(test1_results, test2_results)

    if verbose:
        print("Computing H3 (phenotype matching) statistics...")
    h3_stats = compute_h3_statistics(test1_results)

    if verbose:
        print("Computing Bayesian hypothesis testing statistics...")
    bayesian_stats = compute_bayesian_statistics(test1_results, test2_results, h2_stats)

    # Compile results
    results = {
        'metadata': {
            'n_listeners': n_listeners,
            'seed': seed,
            'use_nhanes_priors': use_nhanes_priors,
            'timestamp': datetime.now().isoformat(),
            'n_phenotypes': len(PHENOTYPE_DEFINITIONS),
            'phenotype_distribution': n_per_phenotype,
        },
        'h1_efficiency': h1_stats,
        'h2_reliability': h2_stats,
        'h3_phenotype_matching': h3_stats,
        'bayesian_analysis': bayesian_stats,
        'population_data': [
            {
                'listener_id': p['listener_id'],
                'phenotype': p['phenotype'],
                'category': p['category'],
                'audiogram': p['audiogram'],
            }
            for p in population
        ],
        'psychometric_params': psych_params_list,
    }

    # Save results
    if verbose:
        print(f"\nSaving results to {output_dir}...")

    # Save summary JSON
    summary_path = output_dir / "stage1_summary.json"
    summary = {
        'metadata': results['metadata'],
        'h1_efficiency': {k: v for k, v in h1_stats.items() if k != 'by_phenotype'},
        'h2_reliability': {k: v for k, v in h2_stats.items()
                          if not k.endswith('_result') and not k.startswith('bayes_test') and not k.startswith('mhw_test')},
        'h3_phenotype_matching': {k: v for k, v in h3_stats.items()
                                  if k not in ['predicted_gains', 'observed_gains', 'phenotypes']},
        'bayesian_analysis': {k: v for k, v in bayesian_stats.items()
                              if k not in ['h1_result', 'h2_result']},  # Exclude full result objects
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Save full results as pickle
    full_path = output_dir / "stage1_full_results.pkl"
    with open(full_path, 'wb') as f:
        pickle.dump(results, f)

    # Print summary
    if verbose:
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)

        print("\nH1: Efficiency")
        print(f"  Bayesian mean trials: {h1_stats['bayes_mean_trials']:.1f} ± {h1_stats['bayes_std_trials']:.1f}")
        print(f"  mHW mean trials: {h1_stats['mhw_mean_trials']:.1f} ± {h1_stats['mhw_std_trials']:.1f}")
        print(f"  Reduction: {h1_stats['reduction_pct']:.1f}%")
        print(f"  Cohen's d: {h1_stats['cohens_d']:.2f}")
        print(f"  p-value: {h1_stats['p_value']:.2e}")

        print("\nH2: Reliability")
        print(f"  ICC (Bayesian): {h2_stats['icc_bayes']:.3f} [{h2_stats['icc_bayes_ci'][0]:.3f}, {h2_stats['icc_bayes_ci'][1]:.3f}]")
        print(f"  ICC (mHW): {h2_stats['icc_mhw']:.3f} [{h2_stats['icc_mhw_ci'][0]:.3f}, {h2_stats['icc_mhw_ci'][1]:.3f}]")
        print(f"  Test-retest SD (Bayesian): {h2_stats['bayes_test_retest_sd']:.2f} dB")
        print(f"  Test-retest SD (mHW): {h2_stats['mhw_test_retest_sd']:.2f} dB")

        print("\nH3: Phenotype Matching")
        print(f"  Matching accuracy: {h3_stats['matching_accuracy']:.1%}")
        print(f"  Correlation (predicted vs observed): {h3_stats['correlation']:.3f}")
        print(f"  Correlation 95% CI: [{h3_stats['correlation_ci'][0]:.3f}, {h3_stats['correlation_ci'][1]:.3f}]")
        print(f"  Correlation p-value: {h3_stats['correlation_p_value']:.2e}")

        print("\n" + "-"*60)
        print("BAYESIAN HYPOTHESIS TESTING")
        print("-"*60)

        print("\nH1: Efficiency (Bayesian)")
        print(f"  Bayes Factor (BF₁₀): {bayesian_stats['h1_bf10']:.2f}")
        print(f"  Interpretation: {bayesian_stats['h1_bf_interpretation']}")
        print(f"  Posterior mean: {bayesian_stats['h1_posterior_mean']:.2f} trials")
        print(f"  95% HDI: [{bayesian_stats['h1_posterior_hdi'][0]:.2f}, {bayesian_stats['h1_posterior_hdi'][1]:.2f}]")
        print(f"  P(δ > {bayesian_stats['h1_practical_threshold']:.0f} trials): {bayesian_stats['h1_prob_practical']:.3f}")

        print("\nH2: Reliability (Bayesian)")
        print(f"  Bayes Factor (BF₁₀): {bayesian_stats['h2_bf10']:.2f}")
        print(f"  Interpretation: {bayesian_stats['h2_bf_interpretation']}")
        print(f"  Posterior mean: {bayesian_stats['h2_posterior_mean']:.2f} dB")
        print(f"  95% HDI: [{bayesian_stats['h2_posterior_hdi'][0]:.2f}, {bayesian_stats['h2_posterior_hdi'][1]:.2f}]")
        print(f"  P(δ > {bayesian_stats['h2_practical_threshold']:.0f} dB): {bayesian_stats['h2_prob_practical']:.3f}")

        print("\nBayesian ICC Estimates")
        print(f"  Bayesian procedure: {bayesian_stats['icc_bayes_bayesian']:.3f} [{bayesian_stats['icc_bayes_bayesian_hdi'][0]:.3f}, {bayesian_stats['icc_bayes_bayesian_hdi'][1]:.3f}]")
        print(f"    P(ICC > 0.9): {bayesian_stats['icc_bayes_prob_excellent']:.3f}")
        print(f"  mHW procedure: {bayesian_stats['icc_mhw_bayesian']:.3f} [{bayesian_stats['icc_mhw_bayesian_hdi'][0]:.3f}, {bayesian_stats['icc_mhw_bayesian_hdi'][1]:.3f}]")
        print(f"    P(ICC > 0.9): {bayesian_stats['icc_mhw_prob_excellent']:.3f}")

        print(f"\nResults saved to: {output_dir}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run Stage 1 manuscript simulation"
    )
    parser.add_argument(
        '--n_listeners', type=int, default=2200,
        help='Number of listeners (default: 2200)'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)'
    )
    parser.add_argument(
        '--output_dir', type=str, default=None,
        help='Output directory'
    )
    parser.add_argument(
        '--mini', action='store_true',
        help='Run mini simulation (50 listeners) for testing'
    )
    parser.add_argument(
        '--nhanes', action='store_true',
        help='Use NHANES priors'
    )
    parser.add_argument(
        '--quiet', action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    n_listeners = 50 if args.mini else args.n_listeners
    output_dir = Path(args.output_dir) if args.output_dir else None

    if args.mini and output_dir is None:
        output_dir = Path("results/stage1_mini")

    run_simulation(
        n_listeners=n_listeners,
        seed=args.seed,
        output_dir=output_dir,
        use_nhanes_priors=args.nhanes,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
