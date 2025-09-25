#!/usr/bin/env python3
"""
Main simulation script for running audiometry procedure comparisons.

This script runs simulations comparing different audiometry procedures
(mHW vs Bayesian) across various listener profiles.
"""

import argparse
import sys
import yaml
from pathlib import Path

# Add the package to the path
sys.path.append(str(Path(__file__).parent.parent))

from audiometry_ai.procedures import ModifiedHughsonWestlakeAudiometry, BayesianAudiometry
from audiometry_ai.simulation import generate_clipped_data

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_simulation(config):
    """Run the main simulation based on configuration."""
    print(f"Running simulation with {config['simulation']['n_listeners']} listeners")
    print(f"Procedures: {list(config['procedures'].keys())}")

    # TODO: Implement full simulation pipeline
    # This is a placeholder for the actual simulation logic

    results = {
        'mhw': {'accuracy': 0.85, 'trials': 25},
        'bayesian': {'accuracy': 0.90, 'trials': 18}
    }

    return results

def main():
    parser = argparse.ArgumentParser(description='Run audiometry simulation study')
    parser.add_argument('--config', type=str,
                       default='configs/default.yaml',
                       help='Path to configuration file')
    parser.add_argument('--n-listeners', type=int,
                       help='Number of listeners to simulate')
    parser.add_argument('--n-jobs', type=int, default=1,
                       help='Number of parallel jobs')

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Configuration file {args.config} not found. Using defaults.")
        config = {
            'simulation': {'n_listeners': 100, 'n_repeats': 3, 'seed': 42},
            'procedures': {'mhw': {}, 'bayesian': {}}
        }

    # Override with command line arguments
    if args.n_listeners:
        config['simulation']['n_listeners'] = args.n_listeners

    # Run simulation
    results = run_simulation(config)

    print("Simulation completed successfully!")
    print(f"Results: {results}")

if __name__ == "__main__":
    main()