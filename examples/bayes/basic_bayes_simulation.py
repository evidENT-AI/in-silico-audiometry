"""Example usage of the Bayesian audiometry testing system with saved visualizations."""

import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.simulations.models.response_model import HearingResponseModel
from src.simulations.models.basic_bayes import BayesianPureToneAudiometry
from src.visualization.simulation_plotting import (
    plot_audiogram,
    plot_psychometric_comparison
)
from src.visualization.bayes_plots import plot_pdf_evolution, plot_pdf_evolution_gaussians, plot_final_pdfs

def print_progression(progression, frequency):
    """Print detailed progression for a specific frequency."""
    print(f"\nProgression for {frequency} Hz:")
    print("Level | Response | Est. Threshold | Uncertainty | Phase")
    print("-" * 65)
    for level, response, threshold, uncertainty, phase in progression:
        print(f"{level:3.0f} dB | {str(response):5} | {threshold:13.1f} | {uncertainty:10.1f} | {phase}")

def save_figure(fig, filename, results_dir):
    """Helper function to properly save figures."""
    fig.savefig(results_dir / filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

def main():
    # Turn off interactive plotting
    plt.ioff()

    # Ensure results directory exists
    results_dir = Path("/Users/liambarrett/Evident-AI/audiometry_ai/results/examples/bayes")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Create simulated hearing profile
    hearing_profile = {
        250: 10,   
        500: 15,   
        1000: 20,  
        2000: 30,  
        4000: 45,  
        8000: 50   
    }

    # Configure response model
    response_params = {
        'slope': 10,      
        'guess_rate': 0.01,  
        'lapse_rate': 0.01,
        'threshold_probability': 0.5
    }

    # Initialize and run test
    audiometry = BayesianPureToneAudiometry(
        hearing_profile_data=hearing_profile,
        response_model_params=response_params,
        fp_rate=0.01,
        fn_rate=0.01,
        random_state=42
    )

    # Get results
    results = audiometry.perform_test()
    thresholds = results['thresholds']
    uncertainties = results['uncertainties']
    progression_patterns = results['progression_patterns']
    pdf_histories = results['pdf_history']

    # Print statistics
    print("\nTest Results:")
    print("\nThresholds (with uncertainties):")
    for freq in sorted(thresholds.keys()):
        print(f"{freq} Hz: {thresholds[freq]:.1f} dB ± {uncertainties[freq]:.1f} dB")

    trials_per_freq = [len(prog) for prog in progression_patterns.values()]
    mean_trials = np.mean(trials_per_freq)
    std_trials = np.std(trials_per_freq)
    
    print("\nTrial Statistics:")
    print(f"Mean trials per frequency: {mean_trials:.1f}")
    print(f"Standard deviation: {std_trials:.1f}")
    print(f"Range: {min(trials_per_freq)} - {max(trials_per_freq)}")

    print_progression(progression_patterns[1000], 1000)

     # Save visualizations
    # Audiogram
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_audiogram(thresholds, "Bayesian Audiometry Results")
    #plt.savefig(results_dir / "audiogram.png", bbox_inches='tight', dpi=300)
    plt.close()

    # PDF evolution plots for 1000 Hz
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_pdf_evolution(
        pdf_histories[1000],
        audiometry.db_range,
        1000,
        progression_patterns[1000]
    )
    #plt.savefig(results_dir / "pdf_evolution_1000hz.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_pdf_evolution_gaussians(
        pdf_histories[1000],
        audiometry.db_range,
        1000,
        progression_patterns[1000],
        n_trials_to_show=10
    )
    #plt.savefig(results_dir / "pdf_evolution_gaussians_1000hz.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Plot final PDFs
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_final_pdfs(
        posteriors={f: pdf_histories[f][-1] for f in thresholds.keys()},
        db_range=audiometry.db_range,
        thresholds=thresholds
    )
    #plt.savefig(results_dir / "final_pdfs.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Psychometric comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_psychometric_comparison(
        HearingResponseModel,
        threshold_probabilities=[0.3, 0.5, 0.7]
    )
    #plt.savefig(results_dir / "psychometric_comparison.png", bbox_inches='tight', dpi=300)
    plt.close()

    # Turn interactive plotting back on at the end
    plt.ion()

if __name__ == "__main__":
    main()