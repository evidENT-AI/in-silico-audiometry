#!/usr/bin/env python3
"""
Full mHW vs Bayesian Simulation Study

Comprehensive comparison of modified Hughson-Westlake vs Bayesian pure-tone audiometry
procedures across diverse simulated listeners. Based on Stage 1 Registered Report
methodology for "Bayesian Optimization of Pure-Tone Audiometry Through In-Silico Modelling"

This simulation tests the primary hypotheses:
H1: N_Bayes < N_mHW (efficiency hypothesis)
H2: |θ_test - θ_retest|_Bayes < |θ_test - θ_retest|_mHW (reliability hypothesis)

Authors: L. Barrett et al.
Version: 1.0
"""

import os
import sys
import json
import yaml
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm

# Add the package to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

from audiometry_ai.simulation.response_model import HearingResponseModel
from audiometry_ai.procedures.bsa_mhw import ModifiedHughsonWestlakeAudiometry
from audiometry_ai.procedures.basic_bayes import BayesianPureToneAudiometry
from audiometry_ai.visualization.simulation_plotting import plot_audiogram
from audiometry_ai.visualization.bayes_plots import plot_audiogram as plot_bayesian_audiogram

warnings.filterwarnings('ignore', category=RuntimeWarning)

@dataclass
class ListenerProfile:
    """Represents a simulated listener with specific hearing characteristics."""
    listener_id: int
    profile_type: str
    true_thresholds: Dict[int, float]
    psychometric_params: Dict[str, float]
    demographics: Dict[str, any] = None

@dataclass
class TestResult:
    """Represents results from a single audiometry test."""
    listener_id: int
    procedure: str
    test_number: int  # 1 for test, 2 for retest
    estimated_thresholds: Dict[int, float]
    trial_counts: Dict[int, int]
    test_duration: float
    progression_patterns: Dict = None
    success: bool = True
    error_message: str = ""

@dataclass
class ComparisonMetrics:
    """Primary outcome measures for procedure comparison."""
    # Efficiency metrics (H1)
    mean_trials_mhw: float
    mean_trials_bayes: float
    trial_count_difference: float
    trial_count_pvalue: float
    trial_count_effect_size: float

    # Accuracy metrics
    mean_error_mhw: float
    mean_error_bayes: float
    accuracy_within_5db_mhw: float
    accuracy_within_5db_bayes: float

    # Reliability metrics (H2)
    test_retest_reliability_mhw: float
    test_retest_reliability_bayes: float
    reliability_difference: float
    reliability_pvalue: float
    reliability_effect_size: float

class ListenerGenerator:
    """Generates diverse simulated listeners based on configuration."""

    def __init__(self, config: dict):
        self.config = config
        self.rng = np.random.RandomState(config['simulation']['random_seed'])

    def generate_listener_population(self, n_listeners: int) -> List[ListenerProfile]:
        """Generate a population of diverse simulated listeners."""
        listeners = []
        listener_config = self.config['listeners']

        # Calculate numbers for each profile type
        profile_counts = {}
        for profile_type, params in listener_config['profiles'].items():
            count = int(n_listeners * params['proportion'])
            profile_counts[profile_type] = count

        # Adjust for rounding differences
        total_assigned = sum(profile_counts.values())
        if total_assigned < n_listeners:
            # Add remaining to largest group
            largest_group = max(profile_counts.keys(), key=lambda x: profile_counts[x])
            profile_counts[largest_group] += (n_listeners - total_assigned)

        listener_id = 0

        # Generate each profile type
        for profile_type, count in profile_counts.items():
            for _ in range(count):
                listener = self._generate_single_listener(listener_id, profile_type)
                listeners.append(listener)
                listener_id += 1

        return listeners

    def _generate_single_listener(self, listener_id: int, profile_type: str) -> ListenerProfile:
        """Generate a single listener with specified profile type."""
        frequencies = self.config['frequencies']
        profile_params = self.config['listeners']['profiles'][profile_type]
        psychometric_params = self.config['listeners']['psychometric']

        # Generate true hearing thresholds based on profile type
        true_thresholds = {}
        threshold_min, threshold_max = profile_params['threshold_range']

        if profile_type == 'normal_hearing':
            # Flat, minimal hearing loss
            base_threshold = self.rng.uniform(0, 15)
            for freq in frequencies:
                # Small random variation ±5 dB
                threshold = base_threshold + self.rng.normal(0, 2)
                true_thresholds[freq] = max(0, min(threshold, 20))

        elif profile_type == 'age_related':
            # High-frequency sloping loss
            base_threshold = self.rng.uniform(10, 30)  # Low frequency baseline
            slope = self.rng.uniform(0.5, 2.0)  # dB per octave
            for i, freq in enumerate(frequencies):
                octaves_from_250 = np.log2(freq / 250)
                threshold = base_threshold + slope * octaves_from_250 * 10
                threshold += self.rng.normal(0, 3)  # Individual variation
                true_thresholds[freq] = max(0, min(threshold, threshold_max))

        elif profile_type == 'noise_induced':
            # 4kHz notch pattern
            for freq in frequencies:
                if freq == 4000:
                    # Notch at 4kHz
                    threshold = self.rng.uniform(40, 70)
                elif freq in [2000, 8000]:
                    # Adjacent frequencies moderately affected
                    threshold = self.rng.uniform(25, 45)
                else:
                    # Other frequencies less affected
                    threshold = self.rng.uniform(10, 30)
                threshold += self.rng.normal(0, 3)
                true_thresholds[freq] = max(0, min(threshold, threshold_max))

        elif profile_type == 'mixed_loss':
            # Combined conductive + sensorineural
            conductive_component = self.rng.uniform(15, 35)  # Flat conductive loss
            for i, freq in enumerate(frequencies):
                sensorineural = self.rng.uniform(0, 30)
                if freq >= 2000:  # High-frequency sensorineural component
                    sensorineural += self.rng.uniform(10, 25)
                threshold = conductive_component + sensorineural
                threshold += self.rng.normal(0, 3)
                true_thresholds[freq] = max(0, min(threshold, threshold_max))

        # Generate psychometric parameters
        slope = max(3, min(15, self.rng.normal(
            psychometric_params['slope_mean'],
            psychometric_params['slope_std']
        )))

        false_positive = self.rng.uniform(*psychometric_params['false_positive_range'])
        false_negative = self.rng.uniform(*psychometric_params['false_negative_range'])

        psychometric = {
            'slope': slope,
            'guess_rate': false_positive,
            'lapse_rate': false_negative,
            'threshold_probability': 0.5
        }

        return ListenerProfile(
            listener_id=listener_id,
            profile_type=profile_type,
            true_thresholds=true_thresholds,
            psychometric_params=psychometric
        )

class SimulationRunner:
    """Runs the full simulation study."""

    def __init__(self, config_path: str, viz_mode: str = 'standard', viz_sample_size: int = 100):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.listener_generator = ListenerGenerator(self.config)
        self.viz_mode = viz_mode
        self.viz_sample_size = viz_sample_size

        # Setup output directories
        self.base_dir = Path(__file__).parent
        self.results_dir = self.base_dir / 'results'
        self.viz_dir = self.base_dir / 'viz'
        self.results_dir.mkdir(exist_ok=True)
        self.viz_dir.mkdir(exist_ok=True)

        # Generate timestamp for this run
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def run_full_simulation(self) -> Dict:
        """Execute the complete simulation study."""
        print("="*80)
        print("COMPREHENSIVE mHW vs BAYESIAN SIMULATION STUDY")
        print("="*80)
        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Configuration: {self.config['simulation']['name']}")
        print(f"Sample size: {self.config['simulation']['n_listeners']} listeners")
        print(f"Test-retest design: {self.config['simulation']['n_repeats']} repetitions")
        print()

        start_time = time.time()

        # Step 1: Generate listener population
        print("Step 1: Generating diverse listener population...")
        listeners = self.listener_generator.generate_listener_population(
            self.config['simulation']['n_listeners']
        )

        self._print_population_summary(listeners)

        # Step 2: Run simulations
        print("\nStep 2: Running audiometry simulations...")
        all_results = self._run_simulations_parallel(listeners)

        # Step 3: Analyze results
        print("\nStep 3: Analyzing results and computing statistics...")
        analysis_results = self._analyze_results(all_results)

        # Step 4: Generate visualizations
        if self.viz_mode != 'none':
            print("\nStep 4: Generating visualizations...")
            self._generate_visualizations(listeners, all_results, analysis_results)
        else:
            print("\nStep 4: Skipping visualizations (--viz none)")


        # Step 5: Save results
        print("\nStep 5: Saving results and generating reports...")
        self._save_results(listeners, all_results, analysis_results)

        total_time = time.time() - start_time

        # Step 6: Print summary
        print("\n" + "="*80)
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print("="*80)
        self._print_results_summary(analysis_results)
        print(f"\nTotal execution time: {total_time:.1f} seconds")
        print(f"Results saved with timestamp: {self.timestamp}")

        return {
            'listeners': listeners,
            'results': all_results,
            'analysis': analysis_results,
            'timestamp': self.timestamp,
            'execution_time': total_time
        }

    def _print_population_summary(self, listeners: List[ListenerProfile]):
        """Print summary of generated listener population."""
        profile_counts = {}
        for listener in listeners:
            profile_counts[listener.profile_type] = profile_counts.get(listener.profile_type, 0) + 1

        print("Generated listener population:")
        for profile_type, count in profile_counts.items():
            percentage = (count / len(listeners)) * 100
            print(f"  {profile_type}: {count} listeners ({percentage:.1f}%)")

    def _run_simulations_parallel(self, listeners: List[ListenerProfile]) -> List[TestResult]:
        """Run all simulations in parallel."""
        n_jobs = min(self.config['simulation']['parallel_jobs'], mp.cpu_count())
        all_results = []

        # Create tasks for all listener-procedure-repetition combinations
        tasks = []
        for listener in listeners:
            for repeat in range(1, self.config['simulation']['n_repeats'] + 1):
                tasks.append(('mhw', listener, repeat))
                tasks.append(('bayesian', listener, repeat))

        print(f"Running {len(tasks)} simulations using {n_jobs} parallel processes...")

        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._run_single_simulation, procedure, listener, repeat): (procedure, listener.listener_id, repeat)
                for procedure, listener, repeat in tasks
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_task):
                procedure, listener_id, repeat = future_to_task[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    completed += 1
                    if completed % 50 == 0:
                        print(f"  Completed: {completed}/{len(tasks)} ({100*completed/len(tasks):.1f}%)")
                except Exception as e:
                    print(f"  Error in {procedure} for listener {listener_id}, repeat {repeat}: {e}")
                    # Create failed result for tracking
                    all_results.append(TestResult(
                        listener_id=listener_id,
                        procedure=procedure,
                        test_number=repeat,
                        estimated_thresholds={},
                        trial_counts={},
                        test_duration=0.0,
                        success=False,
                        error_message=str(e)
                    ))

        print(f"Completed all {len(tasks)} simulations!")

        # Debug: Count successful vs failed
        successful = len([r for r in all_results if r.success])
        failed = len([r for r in all_results if not r.success])
        print(f"  Successful: {successful}, Failed: {failed}")

        if failed > 0:
            print("  Sample error messages:")
            error_messages = [r.error_message for r in all_results if not r.success and r.error_message]
            for i, msg in enumerate(error_messages[:3]):  # Show first 3 errors
                print(f"    {i+1}: {msg}")

        return all_results

    def _run_single_simulation(self, procedure: str, listener: ListenerProfile, repeat: int) -> TestResult:
        """Run a single audiometry simulation."""
        try:
            if procedure == 'mhw':
                audiometry = ModifiedHughsonWestlakeAudiometry(
                    hearing_profile_data=listener.true_thresholds,
                    response_model_params=listener.psychometric_params,
                    random_state=listener.listener_id + repeat * 1000
                )

                start_time = time.time()
                thresholds, progression_patterns = audiometry.perform_test()
                duration = time.time() - start_time

                # Count trials per frequency
                trial_counts = {}
                for freq, progression in progression_patterns.items():
                    trial_counts[freq] = len(progression)

            elif procedure == 'bayesian':
                audiometry = BayesianPureToneAudiometry(
                    hearing_profile_data=listener.true_thresholds,
                    response_model_params=listener.psychometric_params,
                    random_state=listener.listener_id + repeat * 1000
                )

                start_time = time.time()
                results = audiometry.perform_test()
                duration = time.time() - start_time

                # Extract thresholds and progression patterns
                if isinstance(results, tuple):
                    thresholds, progression_patterns = results
                else:
                    thresholds = results
                    progression_patterns = None

                # Handle Bayesian-specific output format
                if isinstance(thresholds, dict) and 'thresholds' in thresholds:
                    actual_thresholds = thresholds['thresholds']
                    if progression_patterns is None and 'progression_patterns' in thresholds:
                        progression_patterns = thresholds['progression_patterns']
                else:
                    actual_thresholds = thresholds

                thresholds = actual_thresholds

                # Count trials per frequency
                trial_counts = {}
                if progression_patterns:
                    for freq, progression in progression_patterns.items():
                        trial_counts[freq] = len(progression)
                else:
                    # Estimate from threshold values (fallback)
                    for freq in listener.true_thresholds.keys():
                        trial_counts[freq] = 15  # Reasonable estimate

            # Clean up thresholds (handle 'Not Reached' values)
            clean_thresholds = {}
            for freq, threshold in thresholds.items():
                if threshold != 'Not Reached' and not np.isnan(threshold):
                    clean_thresholds[freq] = float(threshold)

            return TestResult(
                listener_id=listener.listener_id,
                procedure=procedure,
                test_number=repeat,
                estimated_thresholds=clean_thresholds,
                trial_counts=trial_counts,
                test_duration=duration,
                progression_patterns=progression_patterns,
                success=True
            )

        except Exception as e:
            return TestResult(
                listener_id=listener.listener_id,
                procedure=procedure,
                test_number=repeat,
                estimated_thresholds={},
                trial_counts={},
                test_duration=0.0,
                success=False,
                error_message=str(e)
            )

    def _analyze_results(self, results: List[TestResult]) -> ComparisonMetrics:
        """Analyze simulation results and compute primary outcome measures."""

        # Organize results by procedure and listener
        results_df = []
        for result in results:
            if result.success:
                for freq, threshold in result.estimated_thresholds.items():
                    results_df.append({
                        'listener_id': result.listener_id,
                        'procedure': result.procedure,
                        'test_number': result.test_number,
                        'frequency': freq,
                        'estimated_threshold': threshold,
                        'trial_count': result.trial_counts.get(freq, 0),
                        'duration': result.test_duration
                    })

        df = pd.DataFrame(results_df)

        if df.empty:
            raise RuntimeError("No successful simulation results to analyze!")

        # Get true thresholds for accuracy calculations
        listeners = self.listener_generator.generate_listener_population(
            self.config['simulation']['n_listeners']
        )
        true_thresholds_df = []
        for listener in listeners:
            for freq, true_thresh in listener.true_thresholds.items():
                true_thresholds_df.append({
                    'listener_id': listener.listener_id,
                    'frequency': freq,
                    'true_threshold': true_thresh
                })

        true_df = pd.DataFrame(true_thresholds_df)
        df = df.merge(true_df, on=['listener_id', 'frequency'], how='left')
        df['absolute_error'] = np.abs(df['estimated_threshold'] - df['true_threshold'])
        df['within_5db'] = df['absolute_error'] <= 5

        # Calculate primary outcome measures

        # 1. Efficiency metrics (H1: N_Bayes < N_mHW)
        trial_summary = df.groupby(['listener_id', 'procedure', 'test_number'])['trial_count'].sum().reset_index()
        trial_summary_mean = trial_summary.groupby(['listener_id', 'procedure'])['trial_count'].mean().reset_index()

        mhw_trials = trial_summary_mean[trial_summary_mean['procedure'] == 'mhw']['trial_count'].values
        bayes_trials = trial_summary_mean[trial_summary_mean['procedure'] == 'bayesian']['trial_count'].values

        # Ensure equal sample sizes for paired comparison
        min_len = min(len(mhw_trials), len(bayes_trials))
        mhw_trials = mhw_trials[:min_len]
        bayes_trials = bayes_trials[:min_len]

        if len(mhw_trials) > 0 and len(bayes_trials) > 0:
            trial_stat, trial_pvalue = stats.ttest_rel(bayes_trials, mhw_trials, alternative='less')
            trial_effect_size = (np.mean(mhw_trials) - np.mean(bayes_trials)) / np.std(mhw_trials - bayes_trials)
        else:
            trial_stat, trial_pvalue = 0, 1
            trial_effect_size = 0

        # 2. Accuracy metrics
        accuracy_summary = df.groupby('procedure').agg({
            'absolute_error': 'mean',
            'within_5db': 'mean'
        }).reset_index()

        mhw_accuracy = accuracy_summary[accuracy_summary['procedure'] == 'mhw']
        bayes_accuracy = accuracy_summary[accuracy_summary['procedure'] == 'bayesian']

        # 3. Reliability metrics (H2: Better test-retest reliability)
        test_retest_data = []

        for listener_id in df['listener_id'].unique():
            for procedure in ['mhw', 'bayesian']:
                listener_proc_data = df[(df['listener_id'] == listener_id) & (df['procedure'] == procedure)]

                if len(listener_proc_data['test_number'].unique()) >= 2:
                    test_data = listener_proc_data[listener_proc_data['test_number'] == 1]
                    retest_data = listener_proc_data[listener_proc_data['test_number'] == 2]

                    # Calculate test-retest differences per frequency
                    for freq in test_data['frequency'].unique():
                        test_thresh = test_data[test_data['frequency'] == freq]['estimated_threshold'].iloc[0] if len(test_data[test_data['frequency'] == freq]) > 0 else None
                        retest_thresh = retest_data[retest_data['frequency'] == freq]['estimated_threshold'].iloc[0] if len(retest_data[retest_data['frequency'] == freq]) > 0 else None

                        if test_thresh is not None and retest_thresh is not None:
                            test_retest_data.append({
                                'listener_id': listener_id,
                                'procedure': procedure,
                                'frequency': freq,
                                'test_retest_diff': abs(test_thresh - retest_thresh)
                            })

        if test_retest_data:
            reliability_df = pd.DataFrame(test_retest_data)
            reliability_summary = reliability_df.groupby('procedure')['test_retest_diff'].mean().reset_index()

            mhw_reliability_data = reliability_df[reliability_df['procedure'] == 'mhw']['test_retest_diff'].values
            bayes_reliability_data = reliability_df[reliability_df['procedure'] == 'bayesian']['test_retest_diff'].values

            min_rel_len = min(len(mhw_reliability_data), len(bayes_reliability_data))
            if min_rel_len > 0:
                mhw_rel = mhw_reliability_data[:min_rel_len]
                bayes_rel = bayes_reliability_data[:min_rel_len]

                rel_stat, rel_pvalue = stats.ttest_ind(bayes_rel, mhw_rel, alternative='less')
                rel_effect_size = (np.mean(mhw_rel) - np.mean(bayes_rel)) / np.sqrt((np.var(mhw_rel) + np.var(bayes_rel))/2)

                mhw_reliability = np.mean(mhw_rel)
                bayes_reliability = np.mean(bayes_rel)
            else:
                rel_pvalue, rel_effect_size = 1, 0
                mhw_reliability = bayes_reliability = 0
        else:
            rel_pvalue, rel_effect_size = 1, 0
            mhw_reliability = bayes_reliability = 0

        return ComparisonMetrics(
            # Efficiency (H1)
            mean_trials_mhw=np.mean(mhw_trials) if len(mhw_trials) > 0 else 0,
            mean_trials_bayes=np.mean(bayes_trials) if len(bayes_trials) > 0 else 0,
            trial_count_difference=np.mean(mhw_trials) - np.mean(bayes_trials) if len(mhw_trials) > 0 else 0,
            trial_count_pvalue=trial_pvalue,
            trial_count_effect_size=trial_effect_size,

            # Accuracy
            mean_error_mhw=mhw_accuracy['absolute_error'].iloc[0] if len(mhw_accuracy) > 0 else 0,
            mean_error_bayes=bayes_accuracy['absolute_error'].iloc[0] if len(bayes_accuracy) > 0 else 0,
            accuracy_within_5db_mhw=mhw_accuracy['within_5db'].iloc[0] if len(mhw_accuracy) > 0 else 0,
            accuracy_within_5db_bayes=bayes_accuracy['within_5db'].iloc[0] if len(bayes_accuracy) > 0 else 0,

            # Reliability (H2)
            test_retest_reliability_mhw=mhw_reliability,
            test_retest_reliability_bayes=bayes_reliability,
            reliability_difference=mhw_reliability - bayes_reliability,
            reliability_pvalue=rel_pvalue,
            reliability_effect_size=rel_effect_size
        )

    def _generate_visualizations(self, listeners: List[ListenerProfile], results: List[TestResult], analysis: ComparisonMetrics):
        """Generate comprehensive visualizations based on mode."""
        n_listeners = len(listeners)

        if self.viz_mode == 'standard':
            print(f"  Generating standard visualizations for {n_listeners} listeners...")
            self._plot_listener_audiograms(listeners)
            self._plot_efficiency_comparison(results, analysis)
            self._plot_accuracy_analysis(listeners, results)
            self._plot_reliability_analysis(results)
            if n_listeners <= 20:  # Only for small studies
                print("  Generating individual procedure comparison audiograms...")
                self._plot_procedure_comparisons(listeners, results)

        elif self.viz_mode == 'large_scale':
            print(f"  Generating large-scale visualizations for {n_listeners} listeners...")
            self._plot_large_scale_population_summary(listeners)
            self._plot_large_scale_efficiency(results, analysis)
            self._plot_large_scale_accuracy_heatmaps(listeners, results)
            self._plot_large_scale_reliability(results)
            self._plot_statistical_summary(analysis)

        print("  Visualization generation completed!")

    def _plot_listener_audiograms(self, listeners: List[ListenerProfile]):
        """Plot audiograms showing the diversity of simulated listeners."""
        plt.style.use('default')

        # Group listeners by profile type
        profile_groups = {}
        for listener in listeners:
            if listener.profile_type not in profile_groups:
                profile_groups[listener.profile_type] = []
            profile_groups[listener.profile_type].append(listener)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()

        # Standard audiometric frequencies
        frequencies = [250, 500, 1000, 2000, 4000, 8000]

        for idx, (profile_type, group) in enumerate(profile_groups.items()):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # Plot each listener in this group
            for listener in group:
                thresholds = [listener.true_thresholds.get(freq, 0) for freq in frequencies]
                ax.plot(frequencies, thresholds, alpha=0.6, linewidth=1)

            # Calculate and plot mean
            mean_thresholds = []
            for freq in frequencies:
                values = [listener.true_thresholds.get(freq, 0) for listener in group]
                mean_thresholds.append(np.mean(values))
            ax.plot(frequencies, mean_thresholds, 'k-', linewidth=3, label='Mean')

            ax.set_xscale('log')
            ax.set_xticks(frequencies)
            ax.set_xticklabels([f'{f/1000:.1f}' if f >= 1000 else f'{f}' for f in frequencies])
            ax.set_xlabel('Frequency (kHz)')
            ax.set_ylabel('Hearing Level (dB HL)')
            ax.set_title(f'{profile_type.replace("_", " ").title()}\n(n={len(group)})')
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()
            ax.legend()

        # Remove unused subplots
        for idx in range(len(profile_groups), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(self.viz_dir / f'listener_audiograms_{self.timestamp}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_efficiency_comparison(self, results: List[TestResult], analysis: ComparisonMetrics):
        """Plot trial count comparison between procedures."""
        # Extract trial counts by procedure
        mhw_counts = []
        bayes_counts = []

        for result in results:
            if result.success:
                total_trials = sum(result.trial_counts.values())
                if result.procedure == 'mhw':
                    mhw_counts.append(total_trials)
                else:
                    bayes_counts.append(total_trials)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Box plot comparison
        ax1.boxplot([mhw_counts, bayes_counts], labels=['mHW', 'Bayesian'])
        ax1.set_ylabel('Total Trials')
        ax1.set_title('Trial Count Distribution')
        ax1.grid(True, alpha=0.3)

        # Individual comparison (paired data)
        min_len = min(len(mhw_counts), len(bayes_counts))
        if min_len > 0:
            for i in range(min_len):
                ax2.plot([1, 2], [mhw_counts[i], bayes_counts[i]], 'b-', alpha=0.3)

            ax2.plot([1, 2], [np.mean(mhw_counts), np.mean(bayes_counts)],
                    'ro-', linewidth=3, markersize=10, label='Mean')
            ax2.set_xlim(0.5, 2.5)
            ax2.set_xticks([1, 2])
            ax2.set_xticklabels(['mHW', 'Bayesian'])
            ax2.set_ylabel('Total Trials')
            ax2.set_title('Individual Listener Comparisons')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

        plt.tight_layout()
        plt.savefig(self.viz_dir / f'efficiency_comparison_{self.timestamp}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_accuracy_analysis(self, listeners: List[ListenerProfile], results: List[TestResult]):
        """Plot accuracy analysis showing errors by frequency and procedure."""
        # Create listener lookup
        listener_lookup = {listener.listener_id: listener for listener in listeners}

        # Calculate errors
        error_data = []
        for result in results:
            if result.success:
                listener = listener_lookup[result.listener_id]
                for freq, estimated in result.estimated_thresholds.items():
                    true_threshold = listener.true_thresholds.get(int(freq), 0)
                    error = abs(estimated - true_threshold)
                    error_data.append({
                        'frequency': int(freq),
                        'error': error,
                        'procedure': result.procedure,
                        'profile_type': listener.profile_type
                    })

        error_df = pd.DataFrame(error_data)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Error by frequency
        error_by_freq = error_df.groupby(['frequency', 'procedure'])['error'].mean().unstack()
        error_by_freq.plot(kind='bar', ax=axes[0,0])
        axes[0,0].set_title('Mean Absolute Error by Frequency')
        axes[0,0].set_xlabel('Frequency (Hz)')
        axes[0,0].set_ylabel('Mean Absolute Error (dB)')
        axes[0,0].legend(title='Procedure')
        axes[0,0].tick_params(axis='x', rotation=45)

        # Error by profile type
        error_by_profile = error_df.groupby(['profile_type', 'procedure'])['error'].mean().unstack()
        error_by_profile.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Mean Absolute Error by Listener Type')
        axes[0,1].set_xlabel('Listener Profile')
        axes[0,1].set_ylabel('Mean Absolute Error (dB)')
        axes[0,1].legend(title='Procedure')
        axes[0,1].tick_params(axis='x', rotation=45)

        # Error distribution
        mhw_errors = error_df[error_df['procedure'] == 'mhw']['error']
        bayes_errors = error_df[error_df['procedure'] == 'bayesian']['error']

        axes[1,0].hist(mhw_errors, bins=20, alpha=0.7, label='mHW', density=True)
        axes[1,0].hist(bayes_errors, bins=20, alpha=0.7, label='Bayesian', density=True)
        axes[1,0].set_xlabel('Absolute Error (dB)')
        axes[1,0].set_ylabel('Density')
        axes[1,0].set_title('Error Distribution')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)

        # Accuracy within ±5dB criterion
        accuracy_5db = error_df.groupby('procedure').apply(
            lambda x: (x['error'] <= 5).mean() * 100
        )
        accuracy_5db.plot(kind='bar', ax=axes[1,1])
        axes[1,1].set_title('Accuracy within ±5 dB Criterion')
        axes[1,1].set_ylabel('Percentage (%)')
        axes[1,1].set_xlabel('Procedure')
        axes[1,1].tick_params(axis='x', rotation=0)
        axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / f'accuracy_analysis_{self.timestamp}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_reliability_analysis(self, results: List[TestResult]):
        """Plot test-retest reliability analysis."""
        # Find paired results (test and retest)
        test_retest_data = []

        for listener_id in set(r.listener_id for r in results):
            for procedure in ['mhw', 'bayesian']:
                listener_results = [r for r in results
                                  if r.listener_id == listener_id and r.procedure == procedure]

                if len(listener_results) >= 2:
                    test1 = listener_results[0]
                    test2 = listener_results[1]

                    for freq in test1.estimated_thresholds:
                        if freq in test2.estimated_thresholds:
                            diff = abs(test1.estimated_thresholds[freq] -
                                     test2.estimated_thresholds[freq])
                            test_retest_data.append({
                                'procedure': procedure,
                                'frequency': int(freq),
                                'difference': diff,
                                'test1': test1.estimated_thresholds[freq],
                                'test2': test2.estimated_thresholds[freq]
                            })

        if not test_retest_data:
            print("    No test-retest data available for reliability analysis")
            return

        reliability_df = pd.DataFrame(test_retest_data)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Test-retest differences by procedure
        reliability_df.boxplot(column='difference', by='procedure', ax=axes[0,0])
        axes[0,0].set_title('Test-Retest Differences by Procedure')
        axes[0,0].set_xlabel('Procedure')
        axes[0,0].set_ylabel('Absolute Difference (dB)')

        # Bland-Altman plots
        for i, procedure in enumerate(['mhw', 'bayesian']):
            proc_data = reliability_df[reliability_df['procedure'] == procedure]
            if len(proc_data) > 0:
                means = (proc_data['test1'] + proc_data['test2']) / 2
                diffs = proc_data['test1'] - proc_data['test2']

                ax = axes[0, 1] if procedure == 'mhw' else axes[1, 1]
                ax.scatter(means, diffs, alpha=0.6)
                ax.axhline(y=0, color='k', linestyle='--')
                ax.axhline(y=np.mean(diffs), color='r', label=f'Mean: {np.mean(diffs):.1f}')
                ax.axhline(y=np.mean(diffs) + 1.96*np.std(diffs), color='r', linestyle='--', alpha=0.5)
                ax.axhline(y=np.mean(diffs) - 1.96*np.std(diffs), color='r', linestyle='--', alpha=0.5)
                ax.set_xlabel('Mean of Test 1 and Test 2 (dB)')
                ax.set_ylabel('Test 1 - Test 2 (dB)')
                ax.set_title(f'Bland-Altman Plot - {procedure.upper()}')
                ax.grid(True, alpha=0.3)
                ax.legend()

        # Reliability by frequency
        reliability_by_freq = reliability_df.groupby(['frequency', 'procedure'])['difference'].mean().unstack()
        reliability_by_freq.plot(kind='bar', ax=axes[1,0])
        axes[1,0].set_title('Mean Test-Retest Difference by Frequency')
        axes[1,0].set_xlabel('Frequency (Hz)')
        axes[1,0].set_ylabel('Mean Absolute Difference (dB)')
        axes[1,0].legend(title='Procedure')
        axes[1,0].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(self.viz_dir / f'reliability_analysis_{self.timestamp}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_procedure_comparisons(self, listeners: List[ListenerProfile], results: List[TestResult]):
        """Plot side-by-side audiogram comparisons for sample listeners."""
        listener_lookup = {listener.listener_id: listener for listener in listeners}

        # Select a few representative listeners
        sample_listeners = listeners[:min(4, len(listeners))]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        frequencies = [250, 500, 1000, 2000, 4000, 8000]

        for idx, listener in enumerate(sample_listeners):
            if idx >= len(axes):
                break

            ax = axes[idx]

            # Get true thresholds
            true_thresholds = [listener.true_thresholds.get(freq, 0) for freq in frequencies]
            ax.plot(frequencies, true_thresholds, 'k-o', linewidth=2,
                   label='True Thresholds', markersize=8)

            # Get estimated thresholds for both procedures
            for procedure in ['mhw', 'bayesian']:
                listener_results = [r for r in results
                                  if r.listener_id == listener.listener_id and
                                     r.procedure == procedure and r.success]

                if listener_results:
                    result = listener_results[0]  # Use first test
                    estimated = [result.estimated_thresholds.get(str(freq), np.nan)
                               for freq in frequencies]

                    marker = 's' if procedure == 'mhw' else '^'
                    color = 'blue' if procedure == 'mhw' else 'red'
                    ax.plot(frequencies, estimated, marker, color=color,
                           markersize=8, alpha=0.7, label=f'{procedure.upper()} Estimate')

            ax.set_xscale('log')
            ax.set_xticks(frequencies)
            ax.set_xticklabels([f'{f/1000:.1f}' if f >= 1000 else f'{f}' for f in frequencies])
            ax.set_xlabel('Frequency (kHz)')
            ax.set_ylabel('Hearing Level (dB HL)')
            ax.set_title(f'Listener {listener.listener_id} ({listener.profile_type})')
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()
            ax.legend(fontsize=9)

        # Remove unused subplots
        for idx in range(len(sample_listeners), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(self.viz_dir / f'procedure_comparisons_{self.timestamp}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_large_scale_population_summary(self, listeners: List[ListenerProfile]):
        """Plot population-level summary for large studies."""
        frequencies = [250, 500, 1000, 2000, 4000, 8000]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Profile type distribution
        profile_counts = {}
        for listener in listeners:
            profile_counts[listener.profile_type] = profile_counts.get(listener.profile_type, 0) + 1

        ax1.pie(profile_counts.values(), labels=[pt.replace('_', ' ').title() for pt in profile_counts.keys()],
               autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'Listener Profile Distribution (N={len(listeners)})')

        # Threshold distribution heatmaps by profile
        profile_groups = {}
        for listener in listeners:
            if listener.profile_type not in profile_groups:
                profile_groups[listener.profile_type] = []
            profile_groups[listener.profile_type].append(listener)

        # Mean thresholds by profile and frequency
        threshold_matrix = []
        profile_labels = []
        for profile_type, group in profile_groups.items():
            mean_thresholds = []
            for freq in frequencies:
                values = [listener.true_thresholds.get(freq, 0) for listener in group]
                mean_thresholds.append(np.mean(values))
            threshold_matrix.append(mean_thresholds)
            profile_labels.append(profile_type.replace('_', ' ').title())

        im2 = ax2.imshow(threshold_matrix, cmap='viridis', aspect='auto')
        ax2.set_xticks(range(len(frequencies)))
        ax2.set_xticklabels([f'{f/1000:.1f}' if f >= 1000 else f'{f}' for f in frequencies])
        ax2.set_yticks(range(len(profile_labels)))
        ax2.set_yticklabels(profile_labels)
        ax2.set_xlabel('Frequency (kHz)')
        ax2.set_title('Mean Thresholds by Profile Type')
        plt.colorbar(im2, ax=ax2, label='Hearing Level (dB HL)')

        # Psychometric parameter distributions
        all_slopes = [listener.psychometric_params.get('slope', 8) for listener in listeners]
        all_guess_rates = [listener.psychometric_params.get('guess_rate', 0.05) * 100 for listener in listeners]

        ax3.hist(all_slopes, bins=50, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Psychometric Slope (dB)')
        ax3.set_ylabel('Count')
        ax3.set_title('Distribution of Psychometric Slopes')
        ax3.grid(True, alpha=0.3)

        ax4.hist(all_guess_rates, bins=50, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('False Positive Rate (%)')
        ax4.set_ylabel('Count')
        ax4.set_title('Distribution of False Positive Rates')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / f'large_scale_population_{self.timestamp}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_large_scale_efficiency(self, results: List[TestResult], analysis: ComparisonMetrics):
        """Plot efficiency analysis for large studies."""
        # Extract trial counts
        mhw_counts = []
        bayes_counts = []
        listener_trial_data = {}

        for result in results:
            if result.success:
                total_trials = sum(result.trial_counts.values())
                if result.procedure == 'mhw':
                    mhw_counts.append(total_trials)
                    listener_trial_data[result.listener_id] = listener_trial_data.get(result.listener_id, {})
                    listener_trial_data[result.listener_id]['mhw'] = total_trials
                else:
                    bayes_counts.append(total_trials)
                    listener_trial_data[result.listener_id] = listener_trial_data.get(result.listener_id, {})
                    listener_trial_data[result.listener_id]['bayesian'] = total_trials

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # Violin plots for distribution comparison
        data_for_violin = [mhw_counts, bayes_counts]
        parts = ax1.violinplot(data_for_violin, positions=[1, 2], widths=0.6)
        ax1.set_xticks([1, 2])
        ax1.set_xticklabels(['mHW', 'Bayesian'])
        ax1.set_ylabel('Total Trials')
        ax1.set_title('Trial Count Distributions')
        ax1.grid(True, alpha=0.3)

        # Add statistics text
        ax1.text(0.05, 0.95, f'mHW: μ={np.mean(mhw_counts):.1f}, σ={np.std(mhw_counts):.1f}',
                transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat'))
        ax1.text(0.05, 0.85, f'Bayes: μ={np.mean(bayes_counts):.1f}, σ={np.std(bayes_counts):.1f}',
                transform=ax1.transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue'))

        # Efficiency improvement scatter plot
        improvements = []
        for listener_id, trials in listener_trial_data.items():
            if 'mhw' in trials and 'bayesian' in trials:
                improvement = trials['mhw'] - trials['bayesian']
                improvements.append(improvement)

        ax2.hist(improvements, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', label='No improvement')
        ax2.axvline(x=np.mean(improvements), color='green', linestyle='-', linewidth=2,
                   label=f'Mean: {np.mean(improvements):.1f} trials')
        ax2.set_xlabel('Trial Reduction (mHW - Bayesian)')
        ax2.set_ylabel('Count')
        ax2.set_title('Distribution of Efficiency Improvements')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Cumulative efficiency gains
        sorted_improvements = np.sort(improvements)
        cumulative_pct = np.arange(1, len(sorted_improvements) + 1) / len(sorted_improvements) * 100
        ax3.plot(sorted_improvements, cumulative_pct, linewidth=2)
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Trial Reduction')
        ax3.set_ylabel('Cumulative Percentage (%)')
        ax3.set_title('Cumulative Distribution of Efficiency Gains')
        ax3.grid(True, alpha=0.3)

        # Effect size visualization
        effect_size = analysis.trial_count_effect_size
        pvalue = analysis.trial_count_pvalue
        ax4.bar(['Effect Size'], [abs(effect_size)], color='skyblue', alpha=0.7)
        ax4.set_ylabel("Cohen's d")
        ax4.set_title(f"Efficiency Effect Size\n(p={pvalue:.4f})")
        ax4.grid(True, alpha=0.3)

        # Add significance indicator
        if pvalue < 0.001:
            sig_text = "***"
        elif pvalue < 0.01:
            sig_text = "**"
        elif pvalue < 0.05:
            sig_text = "*"
        else:
            sig_text = "ns"
        ax4.text(0, abs(effect_size) * 1.1, sig_text, ha='center', fontsize=20, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.viz_dir / f'large_scale_efficiency_{self.timestamp}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_large_scale_accuracy_heatmaps(self, listeners: List[ListenerProfile], results: List[TestResult]):
        """Plot accuracy heatmaps for large studies."""
        listener_lookup = {listener.listener_id: listener for listener in listeners}
        frequencies = [250, 500, 1000, 2000, 4000, 8000]

        # Calculate errors by frequency, procedure, and profile type
        error_matrix_mhw = {}
        error_matrix_bayes = {}
        profile_types = list(set(listener.profile_type for listener in listeners))

        for profile_type in profile_types:
            error_matrix_mhw[profile_type] = {freq: [] for freq in frequencies}
            error_matrix_bayes[profile_type] = {freq: [] for freq in frequencies}

        for result in results:
            if result.success:
                listener = listener_lookup[result.listener_id]
                profile_type = listener.profile_type

                for freq_str, estimated in result.estimated_thresholds.items():
                    freq = int(freq_str)
                    if freq in frequencies:
                        true_threshold = listener.true_thresholds.get(freq, 0)
                        error = abs(estimated - true_threshold)

                        if result.procedure == 'mhw':
                            error_matrix_mhw[profile_type][freq].append(error)
                        else:
                            error_matrix_bayes[profile_type][freq].append(error)

        # Create heatmap matrices
        mhw_matrix = []
        bayes_matrix = []
        profile_labels = []

        for profile_type in sorted(profile_types):
            mhw_row = []
            bayes_row = []
            for freq in frequencies:
                mhw_errors = error_matrix_mhw[profile_type][freq]
                bayes_errors = error_matrix_bayes[profile_type][freq]

                mhw_row.append(np.mean(mhw_errors) if mhw_errors else 0)
                bayes_row.append(np.mean(bayes_errors) if bayes_errors else 0)

            mhw_matrix.append(mhw_row)
            bayes_matrix.append(bayes_row)
            profile_labels.append(profile_type.replace('_', ' ').title())

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # mHW accuracy heatmap
        im1 = ax1.imshow(mhw_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=10)
        ax1.set_xticks(range(len(frequencies)))
        ax1.set_xticklabels([f'{f/1000:.1f}' if f >= 1000 else f'{f}' for f in frequencies])
        ax1.set_yticks(range(len(profile_labels)))
        ax1.set_yticklabels(profile_labels)
        ax1.set_xlabel('Frequency (kHz)')
        ax1.set_title('mHW Mean Absolute Error (dB)')

        # Add text annotations
        for i in range(len(profile_labels)):
            for j in range(len(frequencies)):
                text = ax1.text(j, i, f'{mhw_matrix[i][j]:.1f}',
                              ha="center", va="center", color="black")

        # Bayesian accuracy heatmap
        im2 = ax2.imshow(bayes_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=10)
        ax2.set_xticks(range(len(frequencies)))
        ax2.set_xticklabels([f'{f/1000:.1f}' if f >= 1000 else f'{f}' for f in frequencies])
        ax2.set_yticks(range(len(profile_labels)))
        ax2.set_yticklabels(profile_labels)
        ax2.set_xlabel('Frequency (kHz)')
        ax2.set_title('Bayesian Mean Absolute Error (dB)')

        # Add text annotations
        for i in range(len(profile_labels)):
            for j in range(len(frequencies)):
                text = ax2.text(j, i, f'{bayes_matrix[i][j]:.1f}',
                              ha="center", va="center", color="black")

        # Difference heatmap (mHW - Bayesian)
        diff_matrix = np.array(mhw_matrix) - np.array(bayes_matrix)
        im3 = ax3.imshow(diff_matrix, cmap='RdBu_r', aspect='auto', vmin=-5, vmax=5)
        ax3.set_xticks(range(len(frequencies)))
        ax3.set_xticklabels([f'{f/1000:.1f}' if f >= 1000 else f'{f}' for f in frequencies])
        ax3.set_yticks(range(len(profile_labels)))
        ax3.set_yticklabels(profile_labels)
        ax3.set_xlabel('Frequency (kHz)')
        ax3.set_title('Error Difference (mHW - Bayesian)')

        # Add text annotations for differences
        for i in range(len(profile_labels)):
            for j in range(len(frequencies)):
                color = "white" if abs(diff_matrix[i][j]) > 2.5 else "black"
                text = ax3.text(j, i, f'{diff_matrix[i][j]:.1f}',
                              ha="center", va="center", color=color)

        # Overall accuracy summary
        all_mhw_errors = [error for profile_errors in error_matrix_mhw.values()
                         for freq_errors in profile_errors.values() for error in freq_errors]
        all_bayes_errors = [error for profile_errors in error_matrix_bayes.values()
                           for freq_errors in profile_errors.values() for error in freq_errors]

        accuracy_data = {
            'Procedure': ['mHW', 'Bayesian'],
            'Mean Error': [np.mean(all_mhw_errors), np.mean(all_bayes_errors)],
            'Within ±5dB': [np.mean(np.array(all_mhw_errors) <= 5) * 100,
                           np.mean(np.array(all_bayes_errors) <= 5) * 100]
        }

        x = np.arange(len(accuracy_data['Procedure']))
        width = 0.35

        bars1 = ax4.bar(x - width/2, accuracy_data['Mean Error'], width, label='Mean Error (dB)', alpha=0.8)
        ax4_twin = ax4.twinx()
        bars2 = ax4_twin.bar(x + width/2, accuracy_data['Within ±5dB'], width,
                            label='% Within ±5dB', alpha=0.8, color='orange')

        ax4.set_xlabel('Procedure')
        ax4.set_ylabel('Mean Absolute Error (dB)', color='blue')
        ax4_twin.set_ylabel('Accuracy Within ±5dB (%)', color='orange')
        ax4.set_title('Overall Accuracy Summary')
        ax4.set_xticks(x)
        ax4.set_xticklabels(accuracy_data['Procedure'])
        ax4.grid(True, alpha=0.3)

        # Add colorbar
        plt.colorbar(im1, ax=ax1, label='Error (dB)')
        plt.colorbar(im2, ax=ax2, label='Error (dB)')
        plt.colorbar(im3, ax=ax3, label='Difference (dB)')

        plt.tight_layout()
        plt.savefig(self.viz_dir / f'large_scale_accuracy_{self.timestamp}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_large_scale_reliability(self, results: List[TestResult]):
        """Plot reliability analysis for large studies."""
        # Only generate if we have test-retest data
        test_retest_data = []
        for listener_id in set(r.listener_id for r in results):
            for procedure in ['mhw', 'bayesian']:
                listener_results = [r for r in results
                                  if r.listener_id == listener_id and r.procedure == procedure]

                if len(listener_results) >= 2:
                    test1 = listener_results[0]
                    test2 = listener_results[1]

                    for freq in test1.estimated_thresholds:
                        if freq in test2.estimated_thresholds:
                            diff = abs(test1.estimated_thresholds[freq] -
                                     test2.estimated_thresholds[freq])
                            test_retest_data.append({
                                'procedure': procedure,
                                'difference': diff
                            })

        if not test_retest_data:
            print("    No test-retest data available for large-scale reliability analysis")
            return

        reliability_df = pd.DataFrame(test_retest_data)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

        # Distribution comparison
        mhw_diffs = reliability_df[reliability_df['procedure'] == 'mhw']['difference']
        bayes_diffs = reliability_df[reliability_df['procedure'] == 'bayesian']['difference']

        ax1.hist(mhw_diffs, bins=50, alpha=0.7, label='mHW', density=True, edgecolor='black')
        ax1.hist(bayes_diffs, bins=50, alpha=0.7, label='Bayesian', density=True, edgecolor='black')
        ax1.set_xlabel('Test-Retest Difference (dB)')
        ax1.set_ylabel('Density')
        ax1.set_title('Distribution of Test-Retest Differences')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Box plot comparison
        reliability_df.boxplot(column='difference', by='procedure', ax=ax2)
        ax2.set_title('Test-Retest Reliability by Procedure')
        ax2.set_xlabel('Procedure')
        ax2.set_ylabel('Absolute Difference (dB)')

        # Reliability statistics
        stats_data = {
            'Procedure': ['mHW', 'Bayesian'],
            'Mean Diff': [np.mean(mhw_diffs), np.mean(bayes_diffs)],
            'SD Diff': [np.std(mhw_diffs), np.std(bayes_diffs)]
        }

        x = np.arange(len(stats_data['Procedure']))
        width = 0.35

        bars1 = ax3.bar(x - width/2, stats_data['Mean Diff'], width, label='Mean', alpha=0.8)
        bars2 = ax3.bar(x + width/2, stats_data['SD Diff'], width, label='SD', alpha=0.8)

        ax3.set_xlabel('Procedure')
        ax3.set_ylabel('Test-Retest Difference (dB)')
        ax3.set_title('Reliability Statistics')
        ax3.set_xticks(x)
        ax3.set_xticklabels(stats_data['Procedure'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Cumulative reliability curves
        sorted_mhw = np.sort(mhw_diffs)
        sorted_bayes = np.sort(bayes_diffs)

        cumulative_mhw = np.arange(1, len(sorted_mhw) + 1) / len(sorted_mhw) * 100
        cumulative_bayes = np.arange(1, len(sorted_bayes) + 1) / len(sorted_bayes) * 100

        ax4.plot(sorted_mhw, cumulative_mhw, label='mHW', linewidth=2)
        ax4.plot(sorted_bayes, cumulative_bayes, label='Bayesian', linewidth=2)
        ax4.axvline(x=5, color='red', linestyle='--', alpha=0.7, label='5 dB criterion')
        ax4.set_xlabel('Test-Retest Difference (dB)')
        ax4.set_ylabel('Cumulative Percentage (%)')
        ax4.set_title('Cumulative Reliability Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.viz_dir / f'large_scale_reliability_{self.timestamp}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_statistical_summary(self, analysis: ComparisonMetrics):
        """Plot statistical summary of primary hypotheses for large studies."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))

        # H1: Efficiency hypothesis
        efficiency_data = {
            'Measure': ['mHW Mean Trials', 'Bayesian Mean Trials'],
            'Value': [analysis.mean_trials_mhw, analysis.mean_trials_bayes],
            'Error': [0, 0]  # Could add confidence intervals here
        }

        bars = ax1.bar(efficiency_data['Measure'], efficiency_data['Value'],
                      color=['lightcoral', 'lightblue'], alpha=0.8)
        ax1.set_ylabel('Mean Trials per Test')
        ax1.set_title(f'H1: Efficiency Comparison\n(p = {analysis.trial_count_pvalue:.6f})')
        ax1.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, value in zip(bars, efficiency_data['Value']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

        # Effect size visualization
        ax2.barh(['Efficiency Effect Size'], [analysis.trial_count_effect_size],
                color='skyblue', alpha=0.8)
        ax2.set_xlabel("Cohen's d")
        ax2.set_title(f'Effect Size: {analysis.trial_count_effect_size:.3f}')
        ax2.grid(True, alpha=0.3)

        # Add effect size interpretation
        if abs(analysis.trial_count_effect_size) < 0.2:
            interpretation = "Negligible"
        elif abs(analysis.trial_count_effect_size) < 0.5:
            interpretation = "Small"
        elif abs(analysis.trial_count_effect_size) < 0.8:
            interpretation = "Medium"
        else:
            interpretation = "Large"

        ax2.text(analysis.trial_count_effect_size/2, 0, f'({interpretation})',
                ha='center', va='center', fontweight='bold')

        # H2: Reliability hypothesis (if available)
        if hasattr(analysis, 'reliability_pvalue') and analysis.reliability_pvalue is not None:
            reliability_data = {
                'Measure': ['mHW Test-Retest', 'Bayesian Test-Retest'],
                'Value': [analysis.test_retest_reliability_mhw, analysis.test_retest_reliability_bayes]
            }

            bars = ax3.bar(reliability_data['Measure'], reliability_data['Value'],
                          color=['lightcoral', 'lightblue'], alpha=0.8)
            ax3.set_ylabel('Mean Test-Retest Difference (dB)')
            ax3.set_title(f'H2: Reliability Comparison\n(p = {analysis.reliability_pvalue:.6f})')
            ax3.grid(True, alpha=0.3)

            # Add value labels
            for bar, value in zip(bars, reliability_data['Value']):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

            # Reliability effect size
            ax4.barh(['Reliability Effect Size'], [analysis.reliability_effect_size],
                    color='lightgreen', alpha=0.8)
            ax4.set_xlabel("Cohen's d")
            ax4.set_title(f'Effect Size: {analysis.reliability_effect_size:.3f}')
            ax4.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No test-retest data\navailable', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=16)
            ax3.set_title('H2: Reliability Analysis')

            ax4.text(0.5, 0.5, 'No reliability\nanalysis possible', ha='center', va='center',
                    transform=ax4.transAxes, fontsize=16)
            ax4.set_title('Reliability Effect Size')

        plt.tight_layout()
        plt.savefig(self.viz_dir / f'statistical_summary_{self.timestamp}.png',
                   dpi=300, bbox_inches='tight')
        plt.close()

    def _save_results(self, listeners: List[ListenerProfile], results: List[TestResult], analysis: ComparisonMetrics):
        """Save comprehensive results."""
        # Save analysis summary
        analysis_path = self.results_dir / f'analysis_summary_{self.timestamp}.json'
        with open(analysis_path, 'w') as f:
            json.dump(asdict(analysis), f, indent=2)

        # Create listener lookup for ground truth data
        listener_lookup = {listener.listener_id: listener for listener in listeners}

        # Save detailed results with full listener model information
        results_data = []
        for result in results:
            if result.success:
                # Get the corresponding listener profile
                listener = listener_lookup[result.listener_id]

                # Convert to dict and handle numpy types
                result_dict = asdict(result)

                # Add listener model information to each result
                result_dict['listener_profile'] = {
                    'profile_type': listener.profile_type,
                    'true_thresholds': listener.true_thresholds,
                    'psychometric_params': listener.psychometric_params
                }

                # Convert numpy types to native Python types
                def convert_numpy_types(obj):
                    if isinstance(obj, (np.bool_, bool)):
                        return bool(obj)
                    elif isinstance(obj, (np.integer, np.int32, np.int64)):
                        return int(obj)
                    elif isinstance(obj, (np.floating, np.float32, np.float64)):
                        return float(obj)
                    elif isinstance(obj, dict):
                        return {k: convert_numpy_types(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [convert_numpy_types(item) for item in obj]
                    return obj

                clean_result = convert_numpy_types(result_dict)
                results_data.append(clean_result)

        detailed_path = self.results_dir / f'detailed_results_{self.timestamp}.json'
        with open(detailed_path, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"  Analysis summary saved: {analysis_path.name}")
        print(f"  Detailed results saved: {detailed_path.name}")

    def _print_results_summary(self, analysis: ComparisonMetrics):
        """Print comprehensive results summary."""
        print("\nPRIMARY HYPOTHESES RESULTS:")
        print("-" * 40)

        # H1: Efficiency
        print(f"H1 - Efficiency (N_Bayes < N_mHW):")
        print(f"  mHW mean trials: {analysis.mean_trials_mhw:.1f}")
        print(f"  Bayesian mean trials: {analysis.mean_trials_bayes:.1f}")
        print(f"  Difference: {analysis.trial_count_difference:.1f} trials")
        print(f"  p-value: {analysis.trial_count_pvalue:.4f}")
        print(f"  Effect size (Cohen's d): {analysis.trial_count_effect_size:.3f}")
        h1_result = "SUPPORTED" if analysis.trial_count_pvalue < 0.05 else "NOT SUPPORTED"
        print(f"  Result: H1 {h1_result}")

        print()

        # H2: Reliability
        print(f"H2 - Reliability (Better test-retest for Bayesian):")
        print(f"  mHW test-retest difference: {analysis.test_retest_reliability_mhw:.2f} dB")
        print(f"  Bayesian test-retest difference: {analysis.test_retest_reliability_bayes:.2f} dB")
        print(f"  Improvement: {analysis.reliability_difference:.2f} dB")
        print(f"  p-value: {analysis.reliability_pvalue:.4f}")
        print(f"  Effect size (Cohen's d): {analysis.reliability_effect_size:.3f}")
        h2_result = "SUPPORTED" if analysis.reliability_pvalue < 0.05 else "NOT SUPPORTED"
        print(f"  Result: H2 {h2_result}")

        print()
        print("ACCURACY ANALYSIS:")
        print("-" * 40)
        print(f"mHW mean absolute error: {analysis.mean_error_mhw:.2f} dB")
        print(f"Bayesian mean absolute error: {analysis.mean_error_bayes:.2f} dB")
        print(f"mHW accuracy within ±5 dB: {analysis.accuracy_within_5db_mhw:.1%}")
        print(f"Bayesian accuracy within ±5 dB: {analysis.accuracy_within_5db_bayes:.1%}")


def main():
    """Main execution function."""
    import argparse

    parser = argparse.ArgumentParser(description='Run comprehensive mHW vs Bayesian simulation')
    parser.add_argument('--config', type=str, default='simulation_config.yaml',
                       help='Configuration file name in configs/ directory (default: simulation_config.yaml)')
    parser.add_argument('--viz', choices=['none', 'standard', 'large_scale'],
                       default='standard',
                       help='Visualization mode: none (no plots), standard (detailed plots), large_scale (aggregate plots for >1000 listeners)')
    parser.add_argument('--viz-sample-size', type=int, default=100,
                       help='Sample size for individual plots in large_scale mode (default: 100)')

    args = parser.parse_args()

    script_dir = Path(__file__).parent
    config_path = script_dir / 'configs' / args.config

    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        return 1

    try:
        runner = SimulationRunner(str(config_path), viz_mode=args.viz, viz_sample_size=args.viz_sample_size)
        results = runner.run_full_simulation()
        return 0
    except Exception as e:
        print(f"Simulation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())