"""Batch simulation script for analyzing audiometry performance across different hearing loss categories."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import product
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

from src.simulations.models.response_model import HearingResponseModel
from src.simulations.models.expert_mhw import ExpertModifiedHughsonWestlakeAudiometry
from src.simulations.constants.defaults import DEFAULT_TEST_FREQUENCIES

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

class AudiometrySimulator:
    def __init__(self, random_seed: int = 42):
        """Initialize simulator with random seed for reproducibility."""
        self.rng = np.random.RandomState(random_seed)
        self.frequencies = [250, 500, 1000, 2000, 4000, 8000]
        
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

    def run_simulations(self, response_params_list: List[Dict], n_repeats: int = 3) -> pd.DataFrame:
        """Run simulations for all profiles and parameter combinations."""
        # Generate all profiles
        all_profiles = (
            self.generate_flat_profiles() +  # 24 profiles
            self.generate_notch_profiles() + # 24 profiles
            self.generate_sloped_profiles()  # 24 profiles
        )
        
        # Calculate total number of simulations
        n_profiles = len(all_profiles)
        n_params = len(response_params_list)
        total_sims = n_profiles * n_params * n_repeats
        
        print(f"\nSimulation Summary:")
        print(f"Number of unique profiles: {n_profiles}")
        print(f" - Flat profiles: 24")
        print(f" - Notch profiles: 24")
        print(f" - Sloped profiles: 24")
        print(f"Number of parameter combinations: {n_params}")
        print(f"Number of repeats per combination: {n_repeats}")
        print(f"Total number of simulations: {total_sims}")
        
        # Print profile distribution
        profile_types = pd.DataFrame([
            {
                'profile_type': ptype,
                'max_threshold': max(profile.values()),
                'hearing_loss': classify_hearing_loss(max(profile.values()))
            }
            for ptype, profile in all_profiles
        ])
        
        print("\nProfile Distribution:")
        print(profile_types.groupby(['profile_type', 'hearing_loss']).size().unstack(fill_value=0))
        print("\nMaximum thresholds summary:")
        print(profile_types.groupby('profile_type')['max_threshold'].describe())
        
        results = []
        patient_id = 0
        
        # Run simulations with progress tracking
        for (profile_type, profile), params, repeat in product(
            all_profiles, response_params_list, range(n_repeats)
        ):
            patient_id += 1
            max_threshold = max(profile.values())
            hearing_loss_category = classify_hearing_loss(max_threshold)
            
            #if patient_id % 100 == 0:
                #print(f"Processing simulation {patient_id}/{total_sims}")
            
            audiometry = ExpertModifiedHughsonWestlakeAudiometry(
                hearing_profile_data=profile,
                response_model_params=params,
                random_state=self.rng.randint(0, 10000)
            )
            
            thresholds, progression_patterns = audiometry.perform_test()
            
            for freq in self.frequencies:
                if thresholds[freq] != 'Not Reached':
                    n_trials = len(progression_patterns[freq])
                    error = thresholds[freq] - profile[freq]
                    
                    results.append({
                        'patient_id': patient_id,
                        'profile_type': profile_type,
                        'hearing_loss': hearing_loss_category,
                        'frequency': freq,
                        'true_threshold': profile[freq],
                        'estimated_threshold': thresholds[freq],
                        'error': error,
                        'n_trials': n_trials,
                        'slope': params['slope'],
                        'guess_rate': params['guess_rate'],
                        'lapse_rate': params['lapse_rate'],
                        'threshold_probability': params['threshold_probability'],
                        'max_threshold': max_threshold,
                        'repeat': repeat
                    })
        
        return pd.DataFrame(results)

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
    results_df.to_csv('/Users/liambarrett/Evident-AI/audiometry_ai/results/examples/expert_basic_simulation_results.csv', index=False)
    
    # Analyze and plot results
    (overall_stats, hearing_loss_stats), figures = simulator.analyze_results(results_df)
    
    # Print summary statistics
    print("\nOverall Summary Statistics:")
    print(overall_stats)
    print("\nSummary Statistics by Hearing Loss Category:")
    print(hearing_loss_stats)
    
    # Save figures
    for i, fig in enumerate(figures):
        fig.savefig(f'/Users/liambarrett/Evident-AI/audiometry_ai/results/examples/expert_basic_simulation_plot_{i}.png')
        plt.close(fig)

if __name__ == "__main__":
    main()