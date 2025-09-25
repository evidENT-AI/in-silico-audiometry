"""Parallelized batch simulation script for Bayesian audiometry."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from itertools import product
import multiprocessing as mp
from typing import Dict, List, Tuple
from pathlib import Path
import time
import psutil
from functools import partial

from src.simulations.models.basic_bayes import BayesianPureToneAudiometry

def classify_hearing_loss(threshold: float) -> str:
    """Classify hearing loss based on maximum threshold."""
    if threshold < 25:
        return 'Normal'
    elif threshold < 40:
        return 'Mild'
    elif threshold < 70:
        return 'Moderate'
    elif threshold < 90:
        return 'Severe'
    else:
        return 'Profound'

class ParallelBayesianSimulator:
    def __init__(self, random_seed: int = 42, target_runtime_hours: float = 1.0):
        """
        Initialize simulator with parallel processing capabilities.
        
        Args:
            random_seed: Seed for reproducibility
            target_runtime_hours: Desired runtime in hours
        """
        self.rng = np.random.RandomState(random_seed)
        self.frequencies = [250, 500, 1000, 2000, 4000, 8000]
        
        # Parallel processing setup
        self.n_cores = mp.cpu_count()
        self.available_memory = psutil.virtual_memory().available
        self.single_subject_time = 21.83  # seconds
        self.target_runtime = target_runtime_hours * 3600  # convert to seconds
        
        # Calculate optimal batch size
        self._calculate_optimal_workload()
        
    def generate_flat_profiles(self, n_profiles: int = 24) -> List[Dict[int, float]]:
        """
        Generate flat hearing profiles with different base levels.
        Ensures coverage across all hearing loss categories including profound loss above 120 dB.
        """
        profiles = []
        # Generate base levels to cover all categories, including above 120 dB
        base_levels = [
            *np.linspace(10, 20, 4),     # Normal
            *np.linspace(25, 35, 4),     # Mild
            *np.linspace(45, 65, 4),     # Moderate
            *np.linspace(75, 85, 4),     # Severe
            *np.linspace(95, 140, 8)     # Profound (including above 120 dB)
        ]
        
        for base in base_levels:
            # Add small random variations (±5 dB) around base level
            variations = self.rng.uniform(-5, 5, len(self.frequencies))
            profile = {freq: base + var for freq, var in zip(self.frequencies, variations)}
            profiles.append(('flat', profile))
            
        return profiles

    def generate_notch_profiles(self, n_profiles: int = 24) -> List[Dict[int, float]]:
        """
        Generate profiles with notches at different frequencies.
        Creates deeper notches that can exceed 120 dB.
        """
        profiles = []
        base_levels = np.array([20, 40, 60, 80, 100, 120])  # Different starting levels
        notch_depths = np.array([30, 50, 70, 90])           # Different notch depths
        
        # Create combinations manually to ensure exactly 24 profiles
        # We need 12 combinations (which will become 24 profiles with two frequencies each)
        n_combinations = 12
        base_indices = self.rng.choice(len(base_levels), size=n_combinations, replace=True)
        depth_indices = self.rng.choice(len(notch_depths), size=n_combinations, replace=True)
        
        for base_idx, depth_idx in zip(base_indices, depth_indices):
            base_level = base_levels[base_idx]
            depth = notch_depths[depth_idx]
            
            for notch_freq in [2000, 4000]:  # Two frequencies for each combination
                profile = {freq: base_level for freq in self.frequencies}
                profile[notch_freq] = base_level + depth
                profiles.append(('notch', profile))
                    
        return profiles

    def generate_sloped_profiles(self, n_profiles: int = 24) -> List[Dict[int, float]]:
        """
        Generate profiles with increasing slope with frequency.
        Allows for steeper slopes reaching above 120 dB.
        """
        profiles = []
        start_levels = np.array([0, 20, 40, 60, 80, 100])  # Different starting points
        slopes = np.array([10, 20, 30, 40])                # Steeper slopes for more variation
        
        # Create 24 combinations by randomly selecting start levels and slopes
        start_indices = self.rng.choice(len(start_levels), size=n_profiles, replace=True)
        slope_indices = self.rng.choice(len(slopes), size=n_profiles, replace=True)
        
        for start_idx, slope_idx in zip(start_indices, slope_indices):
            start = start_levels[start_idx]
            slope = slopes[slope_idx]
            
            profile = {}
            for i, freq in enumerate(self.frequencies):
                # Calculate level based on slope and frequency position
                level = start + i * slope
                profile[freq] = level
            profiles.append(('slope', profile))
                
        return profiles

    def _calculate_optimal_workload(self):
        """Calculate optimal number of simulations based on constraints."""
        # Estimate parallel efficiency (assuming 80% efficiency due to overhead)
        parallel_efficiency = 0.8
        effective_cores = int(self.n_cores * parallel_efficiency)
        
        # Calculate maximum subjects we can process in target runtime
        max_subjects = int((self.target_runtime * effective_cores) / self.single_subject_time)
        
        print(f"\nParallel Processing Capabilities:")
        print(f"Available CPU cores: {self.n_cores}")
        print(f"Effective parallel cores: {effective_cores}")
        print(f"Target runtime: {self.target_runtime/3600:.1f} hours")
        print(f"Estimated maximum subjects: {max_subjects}")
        
        # Adjust parameter combinations to fit within max_subjects
        self._adjust_parameter_space(max_subjects)

    def _adjust_parameter_space(self, max_subjects: int):
        """Adjust simulation parameters to fit within max_subjects constraint."""
        # Reduced parameter combinations
        self.slopes = [5, 15]
        self.guess_rates = [0.01, 0.1]
        self.lapse_rates = [0.01, 0.1]
        self.thresh_probs = [0.3, 0.7]
        
        # Calculate profiles needed
        param_combinations = (
            len(self.slopes) *
            len(self.guess_rates) *
            len(self.lapse_rates) *
            len(self.thresh_probs)
        )
        
        self.n_repeats = 3
        profiles_per_type = max(1, int(max_subjects / (3 * param_combinations * self.n_repeats)))
        
        print(f"\nAdjusted Parameters:")
        print(f"Profiles per type: {profiles_per_type}")
        print(f"Parameter combinations: {param_combinations}")
        print(f"Repeats: {self.n_repeats}")
        print(f"Total simulations: {profiles_per_type * 3 * param_combinations * self.n_repeats}")
        
        self.profiles_per_type = profiles_per_type

    def _simulate_single_case(self, args):
        """Run single simulation case (for parallel processing)."""
        (profile_type, profile), params, repeat = args
        max_threshold = max(profile.values())
        hearing_loss_category = classify_hearing_loss(max_threshold)
        
        audiometry = BayesianPureToneAudiometry(
            hearing_profile_data=profile,
            response_model_params=params,
            fp_rate=params['guess_rate'],
            fn_rate=params['lapse_rate'],
            random_state=self.rng.randint(0, 10000)
        )
        
        results_dict = audiometry.perform_test()
        
        results = []
        for freq in self.frequencies:
            results.append({
                'profile_type': profile_type,
                'hearing_loss': hearing_loss_category,
                'frequency': freq,
                'true_threshold': profile[freq],
                'estimated_threshold': results_dict['thresholds'][freq],
                'uncertainty': results_dict['uncertainties'][freq],
                'error': results_dict['thresholds'][freq] - profile[freq],
                'n_trials': len(results_dict['progression_patterns'][freq]),
                **params,
                'max_threshold': max_threshold,
                'repeat': repeat
            })
        
        return results

    def run_parallel_simulations(self) -> pd.DataFrame:
        """Run simulations in parallel with progress bar."""
        # Generate all test cases
        all_profiles = (
            self.generate_flat_profiles(self.profiles_per_type) +
            self.generate_notch_profiles(self.profiles_per_type) +
            self.generate_sloped_profiles(self.profiles_per_type)
        )
        
        response_params_list = [
            {
                'slope': slope,
                'guess_rate': guess,
                'lapse_rate': lapse,
                'threshold_probability': thresh_prob
            }
            for slope, guess, lapse, thresh_prob in product(
                self.slopes,
                self.guess_rates,
                self.lapse_rates,
                self.thresh_probs
            )
        ]
        
        # Create all combinations
        all_cases = list(product(all_profiles, response_params_list, range(self.n_repeats)))
        total_cases = len(all_cases)
        
        print(f"\nStarting simulations:")
        print(f"Total cases to process: {total_cases}")
        print(f"Using {self.n_cores} CPU cores")
        
        # Run parallel simulations with progress bar
        start_time = time.time()
        with mp.Pool(processes=self.n_cores) as pool:
            results = list(tqdm(
                pool.imap(self._simulate_single_case, all_cases),
                total=total_cases,
                desc="Processing subjects",
                unit="subject"
            ))
        end_time = time.time()
        
        # Flatten results
        flat_results = [item for sublist in results for item in sublist]
        
        total_time = end_time - start_time
        print(f"\nSimulation Complete:")
        print(f"Total runtime: {total_time/3600:.2f} hours")
        print(f"Average time per subject: {total_time/total_cases:.2f} seconds")
        print(f"Effective processing rate: {total_cases/total_time:.2f} subjects/second")
        print(f"Parallel efficiency: {(total_cases * self.single_subject_time)/(total_time * self.n_cores):.1%}")
        
        return pd.DataFrame(flat_results)

    def analyze_results(self, df: pd.DataFrame) -> Tuple[Dict, List[plt.Figure]]:
        """Analyze simulation results and create visualizations."""
        figures = []
        
        # 1. Overall summary statistics
        overall_stats = df.groupby('profile_type').agg({
            'error': ['mean', 'std', 'count'],
            'n_trials': ['mean', 'std']
        }).round(2)
        
        # 2. Summary statistics by hearing loss category
        hearing_loss_stats = df.groupby(['hearing_loss', 'profile_type']).agg({
            'error': ['mean', 'std', 'count'],
            'n_trials': ['mean', 'std']
        }).round(2)
        
        # 3. Plot error distributions by hearing loss category
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x='hearing_loss', y='error', hue='profile_type')
        ax.set_title('Error Distribution by Hearing Loss Category and Profile Type')
        figures.append(fig)
        
        # 4. Plot number of trials by hearing loss category
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x='hearing_loss', y='n_trials', hue='profile_type')
        ax.set_title('Number of Trials by Hearing Loss Category and Profile Type')
        figures.append(fig)
        
        # 5. Plot error vs psychometric function parameters by hearing loss category
        param_cols = ['slope', 'threshold_probability', 'guess_rate', 'lapse_rate']
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.ravel()
        
        for i, param in enumerate(param_cols):
            sns.scatterplot(
                data=df, 
                x=param, 
                y='error', 
                hue='hearing_loss', 
                style='profile_type',
                ax=axes[i]
            )
            axes[i].set_title(f'Error vs {param} by Hearing Loss Category')
        
        figures.append(fig)
        
        return (overall_stats, hearing_loss_stats), figures

def main():
    # Initialize simulator with 1-hour target runtime
    simulator = ParallelBayesianSimulator(random_seed=42, target_runtime_hours=1.0)
    
    # Run parallel simulations
    results_df = simulator.run_parallel_simulations()
    
    # Save and analyze results
    results_df.to_csv('/Users/liambarrett/Evident-AI/audiometry_ai/results/examples/bayes/bayes_parallel_simulation_results.csv', index=False)
    
    (overall_stats, hearing_loss_stats), figures = simulator.analyze_results(results_df)
    
    print("\nOverall Summary Statistics:")
    print(overall_stats)
    print("\nSummary Statistics by Hearing Loss Category:")
    print(hearing_loss_stats)
    
    # Save figures
    for i, fig in enumerate(figures):
        fig.savefig(f'/Users/liambarrett/Evident-AI/audiometry_ai/results/examples/bayes/bayes_parallel_simulation_plot_{i}.png')
        plt.close(fig)

if __name__ == "__main__":
    main()