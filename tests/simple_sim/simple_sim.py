#!/usr/bin/env python3
"""
Simple simulation test for audiometry procedures.

This test runs both mHW and Bayesian procedures on a known hearing profile,
saves the results data, and generates visualizations.
"""

import os
import sys
import json
import csv
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

# Add the package to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from audiometry_ai.simulation.response_model import HearingResponseModel
from audiometry_ai.procedures.bsa_mhw import ModifiedHughsonWestlakeAudiometry
from audiometry_ai.procedures.basic_bayes import BayesianPureToneAudiometry
from audiometry_ai.visualization.simulation_plotting import plot_audiogram

def setup_directories():
    """Create results and viz directories if they don't exist."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results')
    viz_dir = os.path.join(script_dir, 'viz')

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(viz_dir, exist_ok=True)

    return results_dir, viz_dir

def save_results_json(data, filepath):
    """Save results data as JSON."""
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(item) for item in obj]
        return obj

    data_serializable = convert_numpy(data)

    with open(filepath, 'w') as f:
        json.dump(data_serializable, f, indent=2)

def save_results_csv(data, filepath):
    """Save threshold results as CSV."""
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Frequency_Hz', 'True_Threshold_dB', 'Estimated_Threshold_dB', 'Error_dB'])

        for freq, estimated in data['estimated_thresholds'].items():
            if isinstance(freq, (int, float)) and estimated != 'Not Reached':
                true_val = data['true_thresholds'].get(freq, 'N/A')
                error = abs(true_val - estimated) if true_val != 'N/A' else 'N/A'
                writer.writerow([freq, true_val, estimated, error])

def create_comparison_plot(mhw_results, bayes_results, true_thresholds, save_path):
    """Create comparison plot of all three threshold sets."""
    frequencies = sorted(true_thresholds.keys())

    true_vals = [true_thresholds[f] for f in frequencies]
    mhw_vals = [mhw_results['estimated_thresholds'].get(f, None) for f in frequencies]
    bayes_vals = []

    # Extract Bayesian thresholds
    if 'thresholds' in bayes_results['estimated_thresholds']:
        bayes_thresh = bayes_results['estimated_thresholds']['thresholds']
        bayes_vals = [bayes_thresh.get(f, None) for f in frequencies]
    else:
        bayes_vals = [bayes_results['estimated_thresholds'].get(f, None) for f in frequencies]

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot true thresholds
    plt.plot(frequencies, true_vals, 'g-o', linewidth=2, markersize=8, label='True Thresholds')

    # Plot mHW results
    mhw_plot_vals = [v if v is not None and v != 'Not Reached' else None for v in mhw_vals]
    valid_freqs_mhw = [f for f, v in zip(frequencies, mhw_plot_vals) if v is not None]
    valid_vals_mhw = [v for v in mhw_plot_vals if v is not None]
    plt.plot(valid_freqs_mhw, valid_vals_mhw, 'b-s', linewidth=2, markersize=8, label='mHW Estimates')

    # Plot Bayesian results
    bayes_plot_vals = [v if v is not None and v != 'Not Reached' else None for v in bayes_vals]
    valid_freqs_bayes = [f for f, v in zip(frequencies, bayes_plot_vals) if v is not None]
    valid_vals_bayes = [v for v in bayes_plot_vals if v is not None]
    plt.plot(valid_freqs_bayes, valid_vals_bayes, 'r-^', linewidth=2, markersize=8, label='Bayesian Estimates')

    plt.xscale('log')
    plt.xticks(frequencies, [f'{f}' for f in frequencies])
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Hearing Level (dB HL)', fontsize=12)
    plt.title('Audiometry Procedure Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()

    # Add error bars for Bayesian uncertainties if available
    if ('uncertainties' in bayes_results['estimated_thresholds'] and
        bayes_results['estimated_thresholds']['uncertainties']):
        uncertainties = bayes_results['estimated_thresholds']['uncertainties']
        bayes_errors = [uncertainties.get(f, 0) for f in valid_freqs_bayes]
        plt.errorbar(valid_freqs_bayes, valid_vals_bayes, yerr=bayes_errors,
                    fmt='none', ecolor='red', alpha=0.5, capsize=3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_error_analysis_plot(mhw_results, bayes_results, true_thresholds, save_path):
    """Create error analysis plot."""
    frequencies = sorted(true_thresholds.keys())

    mhw_errors = []
    bayes_errors = []

    for freq in frequencies:
        true_val = true_thresholds[freq]

        # mHW errors
        mhw_est = mhw_results['estimated_thresholds'].get(freq, None)
        if mhw_est is not None and mhw_est != 'Not Reached':
            mhw_errors.append(abs(true_val - mhw_est))
        else:
            mhw_errors.append(None)

        # Bayesian errors
        if 'thresholds' in bayes_results['estimated_thresholds']:
            bayes_thresh = bayes_results['estimated_thresholds']['thresholds']
            bayes_est = bayes_thresh.get(freq, None)
        else:
            bayes_est = bayes_results['estimated_thresholds'].get(freq, None)

        if bayes_est is not None and bayes_est != 'Not Reached':
            bayes_errors.append(abs(true_val - bayes_est))
        else:
            bayes_errors.append(None)

    # Create error plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Individual frequency errors
    x_pos = np.arange(len(frequencies))
    valid_mhw = [e for e in mhw_errors if e is not None]
    valid_bayes = [e for e in bayes_errors if e is not None]

    valid_freqs_idx_mhw = [i for i, e in enumerate(mhw_errors) if e is not None]
    valid_freqs_idx_bayes = [i for i, e in enumerate(bayes_errors) if e is not None]

    ax1.bar([x - 0.2 for x in valid_freqs_idx_mhw], valid_mhw, 0.4, label='mHW', alpha=0.7)
    ax1.bar([x + 0.2 for x in valid_freqs_idx_bayes], valid_bayes, 0.4, label='Bayesian', alpha=0.7)

    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Absolute Error (dB)')
    ax1.set_title('Error by Frequency')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([f'{f}' for f in frequencies])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Summary statistics
    methods = ['mHW', 'Bayesian']
    mean_errors = [np.mean(valid_mhw) if valid_mhw else 0,
                   np.mean(valid_bayes) if valid_bayes else 0]

    bars = ax2.bar(methods, mean_errors, alpha=0.7, color=['blue', 'red'])
    ax2.set_ylabel('Mean Absolute Error (dB)')
    ax2.set_title('Overall Accuracy Comparison')
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, mean_errors):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.1f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def run_mhw_simulation():
    """Run mHW simulation and return results."""
    print("Running Modified Hughson-Westlake simulation...")

    # Create simulated hearing profile (mild to moderate hearing loss)
    hearing_profile = {
        250: 10,
        500: 15,
        1000: 20,
        2000: 30,
        4000: 45,
        8000: 50
    }

    # Configure response model parameters
    response_params = {
        'slope': 10,
        'guess_rate': 0.02,
        'lapse_rate': 0.02,
        'threshold_probability': 0.5
    }

    try:
        # Initialize audiometry test
        audiometry = ModifiedHughsonWestlakeAudiometry(
            hearing_profile_data=hearing_profile,
            response_model_params=response_params,
            random_state=42
        )

        # Run test
        thresholds, progression_patterns = audiometry.perform_test()

        # Calculate accuracy metrics
        total_error = 0
        freq_count = 0
        errors = {}

        for freq in hearing_profile:
            if freq in thresholds and thresholds[freq] != 'Not Reached':
                error = abs(hearing_profile[freq] - thresholds[freq])
                errors[freq] = error
                total_error += error
                freq_count += 1

        mean_error = total_error / freq_count if freq_count > 0 else float('inf')

        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Modified Hughson-Westlake',
            'true_thresholds': hearing_profile,
            'estimated_thresholds': thresholds,
            'progression_patterns': progression_patterns,
            'response_params': response_params,
            'accuracy_metrics': {
                'mean_absolute_error': mean_error,
                'individual_errors': errors,
                'successful_frequencies': freq_count,
                'total_frequencies': len(hearing_profile)
            },
            'success': True
        }

        print(f"✓ mHW simulation completed successfully!")
        print(f"  Mean absolute error: {mean_error:.1f} dB HL")

        return results

    except Exception as e:
        print(f"✗ mHW simulation failed: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'method': 'Modified Hughson-Westlake'
        }

def run_bayesian_simulation():
    """Run Bayesian simulation and return results."""
    print("Running Bayesian simulation...")

    # Create simulated hearing profile
    hearing_profile = {
        250: 10,
        500: 15,
        1000: 20,
        2000: 30,
        4000: 45,
        8000: 50
    }

    # Configure response model parameters
    response_params = {
        'slope': 10,
        'guess_rate': 0.02,
        'lapse_rate': 0.02,
        'threshold_probability': 0.5
    }

    try:
        # Initialize Bayesian audiometry test
        audiometry = BayesianPureToneAudiometry(
            hearing_profile_data=hearing_profile,
            response_model_params=response_params,
            random_state=42
        )

        # Run test
        simulation_results = audiometry.perform_test()

        # Extract thresholds from results
        if isinstance(simulation_results, tuple):
            thresholds, progression_patterns = simulation_results
        else:
            thresholds = simulation_results
            progression_patterns = None

        # Calculate accuracy metrics
        if 'thresholds' in thresholds:
            estimated_thresholds = thresholds['thresholds']
        else:
            estimated_thresholds = thresholds

        total_error = 0
        freq_count = 0
        errors = {}

        for freq in hearing_profile:
            if freq in estimated_thresholds and estimated_thresholds[freq] != 'Not Reached':
                error = abs(hearing_profile[freq] - estimated_thresholds[freq])
                errors[freq] = error
                total_error += error
                freq_count += 1

        mean_error = total_error / freq_count if freq_count > 0 else float('inf')

        results = {
            'timestamp': datetime.now().isoformat(),
            'method': 'Bayesian Pure-Tone Audiometry',
            'true_thresholds': hearing_profile,
            'estimated_thresholds': thresholds,
            'progression_patterns': progression_patterns,
            'response_params': response_params,
            'accuracy_metrics': {
                'mean_absolute_error': mean_error,
                'individual_errors': errors,
                'successful_frequencies': freq_count,
                'total_frequencies': len(hearing_profile)
            },
            'success': True
        }

        print(f"✓ Bayesian simulation completed successfully!")
        print(f"  Mean absolute error: {mean_error:.1f} dB HL")

        return results

    except Exception as e:
        print(f"✗ Bayesian simulation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'method': 'Bayesian Pure-Tone Audiometry'
        }

def main():
    """Main function to run simulations and generate outputs."""
    print("=" * 60)
    print("SIMPLE AUDIOMETRY SIMULATION TEST")
    print("=" * 60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Setup directories
    results_dir, viz_dir = setup_directories()
    print(f"Results will be saved to: {results_dir}")
    print(f"Visualizations will be saved to: {viz_dir}")
    print()

    # Run simulations
    mhw_results = run_mhw_simulation()
    print()
    bayes_results = run_bayesian_simulation()
    print()

    # Save results
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    if mhw_results['success']:
        mhw_json_path = os.path.join(results_dir, f'mhw_results_{timestamp_str}.json')
        mhw_csv_path = os.path.join(results_dir, f'mhw_thresholds_{timestamp_str}.csv')

        save_results_json(mhw_results, mhw_json_path)
        save_results_csv(mhw_results, mhw_csv_path)
        print(f"✓ mHW results saved: {os.path.basename(mhw_json_path)}, {os.path.basename(mhw_csv_path)}")

    if bayes_results['success']:
        bayes_json_path = os.path.join(results_dir, f'bayes_results_{timestamp_str}.json')
        bayes_csv_path = os.path.join(results_dir, f'bayes_thresholds_{timestamp_str}.csv')

        save_results_json(bayes_results, bayes_json_path)
        save_results_csv(bayes_results, bayes_csv_path)
        print(f"✓ Bayesian results saved: {os.path.basename(bayes_json_path)}, {os.path.basename(bayes_csv_path)}")

    # Generate visualizations
    if mhw_results['success'] and bayes_results['success']:
        print("\nGenerating visualizations...")

        # Comparison plot
        comparison_path = os.path.join(viz_dir, f'threshold_comparison_{timestamp_str}.png')
        create_comparison_plot(mhw_results, bayes_results, mhw_results['true_thresholds'], comparison_path)
        print(f"✓ Comparison plot saved: {os.path.basename(comparison_path)}")

        # Error analysis plot
        error_path = os.path.join(viz_dir, f'error_analysis_{timestamp_str}.png')
        create_error_analysis_plot(mhw_results, bayes_results, mhw_results['true_thresholds'], error_path)
        print(f"✓ Error analysis plot saved: {os.path.basename(error_path)}")

    # Summary
    print("\n" + "=" * 60)
    print("SIMULATION SUMMARY")
    print("=" * 60)

    if mhw_results['success']:
        print(f"mHW: Mean error = {mhw_results['accuracy_metrics']['mean_absolute_error']:.1f} dB HL")
    else:
        print("mHW: Failed")

    if bayes_results['success']:
        print(f"Bayesian: Mean error = {bayes_results['accuracy_metrics']['mean_absolute_error']:.1f} dB HL")
    else:
        print("Bayesian: Failed")

    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if mhw_results['success'] and bayes_results['success']:
        print("\n✓ All tests completed successfully!")
        return 0
    else:
        print("\n✗ Some tests failed.")
        return 1

if __name__ == "__main__":
    exit(main())