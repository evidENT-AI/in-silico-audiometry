"""Batch simulation script for analyzing audiometry performance across different conditions."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

from src.simulations.models.response_model import HearingResponseModel
from src.simulations.models.bsa_mhw import ModifiedHughsonWestlakeAudiometry
from src.simulations.constants.defaults import DEFAULT_TEST_FREQUENCIES

class AudiometrySimulator:
    def __init__(self, random_seed: int = 42):
        """Initialize simulator with random seed for reproducibility."""
        self.rng = np.random.RandomState(random_seed)
        self.frequencies = [250, 500, 1000, 2000, 4000, 8000]
        
    def generate_flat_profiles(self, n_profiles: int = 5) -> List[Dict[int, float]]:
        """Generate flat hearing profiles with different base levels."""
        profiles = []
        base_levels = np.linspace(10, 50, n_profiles)
        
        for base in base_levels:
            # Add small random variations (±5 dB) around base level
            variations = self.rng.uniform(-5, 5, len(self.frequencies))
            profile = {freq: base + var for freq, var in zip(self.frequencies, variations)}
            profiles.append(('flat', profile))
            
        return profiles
    
    def generate_notch_profiles(self, n_profiles: int = 5) -> List[Dict[int, float]]:
        """Generate profiles with notches at different frequencies."""
        profiles = []
        base_level = 20
        notch_depths = np.linspace(20, 40, n_profiles)
        
        for depth in notch_depths:
            for notch_freq in [2000, 4000]:  # Common notch frequencies
                profile = {freq: base_level for freq in self.frequencies}
                profile[notch_freq] = base_level + depth
                profiles.append(('notch', profile))
                
        return profiles
    
    def generate_sloped_profiles(self, n_profiles: int = 5) -> List[Dict[int, float]]:
        """Generate profiles with increasing slope with frequency."""
        profiles = []
        slopes = np.linspace(5, 15, n_profiles)  # dB/octave
        
        for slope in slopes:
            profile = {}
            for i, freq in enumerate(self.frequencies):
                # Calculate level based on slope and frequency position
                profile[freq] = 10 + i * slope
            profiles.append(('slope', profile))
            
        return profiles

    def run_simulations(self, response_params_list: List[Dict], n_repeats: int = 3) -> pd.DataFrame:
        """Run simulations for all profiles and parameter combinations."""
        # Generate all profiles
        all_profiles = (
            self.generate_flat_profiles() +
            self.generate_notch_profiles() +
            self.generate_sloped_profiles()
        )
        
        results = []
        
        # Run simulations for all combinations
        for (profile_type, profile), params, repeat in product(
            all_profiles, response_params_list, range(n_repeats)
        ):
            audiometry = ModifiedHughsonWestlakeAudiometry(
                hearing_profile_data=profile,
                response_model_params=params,
                random_state=self.rng.randint(0, 10000)
            )
            
            thresholds, progression_patterns = audiometry.perform_test()
            
            # Process results for each frequency
            for freq in self.frequencies:
                if thresholds[freq] != 'Not Reached':
                    n_trials = len(progression_patterns[freq])
                    error = thresholds[freq] - profile[freq]
                    
                    results.append({
                        'profile_type': profile_type,
                        'frequency': freq,
                        'true_threshold': profile[freq],
                        'estimated_threshold': thresholds[freq],
                        'error': error,
                        'n_trials': n_trials,
                        'slope': params['slope'],
                        'guess_rate': params['guess_rate'],
                        'lapse_rate': params['lapse_rate'],
                        'threshold_probability': params['threshold_probability'],
                        'repeat': repeat
                    })
        
        return pd.DataFrame(results)

    def analyze_results(self, df: pd.DataFrame) -> Tuple[Dict, List[plt.Figure]]:
        """Analyze simulation results and create visualizations."""
        figures = []
        
        # 1. Summary statistics by profile type
        summary_stats = df.groupby('profile_type').agg({
            'error': ['mean', 'std'],
            'n_trials': ['mean', 'std']
        }).round(2)
        
        # 2. Plot error distributions by profile type
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=df, x='profile_type', y='error')
        ax.set_title('Error Distribution by Profile Type')
        figures.append(fig)
        
        # 3. Plot number of trials by profile type and frequency
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.boxplot(data=df, x='frequency', y='n_trials', hue='profile_type')
        ax.set_title('Number of Trials by Frequency and Profile Type')
        figures.append(fig)
        
        # 4. Plot error vs psychometric function parameters
        param_cols = ['slope', 'threshold_probability', 'guess_rate', 'lapse_rate']
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.ravel()
        
        for i, param in enumerate(param_cols):
            sns.scatterplot(data=df, x=param, y='error', hue='profile_type', ax=axes[i])
            axes[i].set_title(f'Error vs {param}')
        
        figures.append(fig)
        
        return summary_stats, figures

def main():
    # Initialize simulator
    simulator = AudiometrySimulator(random_seed=42)
    
    # Define parameter combinations to test
    response_params_list = [
        {
            'slope': slope,
            'guess_rate': guess,
            'lapse_rate': lapse,
            'threshold_probability': thresh_prob
        }
        for slope, guess, lapse, thresh_prob in product(
            [5, 10, 15],                # slopes
            [0.0, 0.01, 0.1, 0.2],      # guess rates
            [0.0, 0.01, 0.1, 0.2],      # lapse rates
            [0.3, 0.5, 0.7]             # threshold probabilities
        )
    ]
    
    # Run simulations
    results_df = simulator.run_simulations(response_params_list)
    
    # Save raw results
    results_df.to_csv('/Users/liambarrett/Evident-AI/audiometry_ai/results/examples/basic_simulation_results.csv', index=False)
    
    # Analyze and plot results
    summary_stats, figures = simulator.analyze_results(results_df)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(summary_stats)
    
    # Save figures
    for i, fig in enumerate(figures):
        fig.savefig(f'/Users/liambarrett/Evident-AI/audiometry_ai/results/examples/basic_simulation_plot_{i}.png')
        plt.close(fig)

if __name__ == "__main__":
    main()