#!/usr/bin/env python3
"""
Build NHANES-derived priors for Bayesian audiometry.

This script:
1. Downloads required NHANES datasets (audiometry, demographics, questionnaires)
2. Merges and preprocesses the data
3. Builds stratified KDE priors for threshold estimation
4. Saves priors and summary statistics

Usage:
------
    python scripts/build_nhanes_priors.py [--download] [--years 2015-2016 2017-2018]

Arguments:
    --download: Download fresh NHANES data (skip if already downloaded)
    --years: Specific year cycles to use (default: 2015-2016, 2017-2018)
    --output-dir: Output directory for priors (default: data/nhanes)
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from audiometry_ai.data import NHANESDownloader, RECOMMENDED_CYCLES
from audiometry_ai.priors import NHANESPriorBuilder


def download_nhanes_data(output_dir: Path, years: list) -> None:
    """Download required NHANES datasets."""
    print("\n" + "="*60)
    print("DOWNLOADING NHANES DATA")
    print("="*60)

    # Download audiometry data (Examination component)
    print("\n--- Downloading Audiometry Data ---")
    exam_downloader = NHANESDownloader(
        output_dir=output_dir,
        component="Examination"
    )
    exam_downloader.download_datasets(
        datasets=["Audiometry", "Audiometry - Tympanometry"],
        years=years
    )

    # Download demographics data
    print("\n--- Downloading Demographics Data ---")
    demo_downloader = NHANESDownloader(
        output_dir=output_dir,
        component="Demographics"
    )
    demo_downloader.download_datasets(years=years)

    # Download questionnaire data for covariates
    print("\n--- Downloading Questionnaire Data ---")
    quest_downloader = NHANESDownloader(
        output_dir=output_dir,
        component="Questionnaire"
    )
    quest_downloader.download_datasets(
        datasets=["Diabetes", "Medical Conditions", "Blood Pressure & Cholesterol"],
        years=years
    )

    print("\n✅ NHANES data download complete!")


def build_priors(data_dir: Path) -> None:
    """Build KDE priors from downloaded NHANES data."""
    print("\n" + "="*60)
    print("BUILDING NHANES PRIORS")
    print("="*60)

    builder = NHANESPriorBuilder(data_dir=data_dir)

    # Load and merge data
    print("\n--- Loading Data ---")
    builder.load_data()

    # Build all priors
    print("\n--- Building Priors ---")
    priors = builder.build_all_priors()

    # Print summary
    print("\n" + "="*60)
    print("PRIOR CONSTRUCTION SUMMARY")
    print("="*60)

    if 'marginal' in priors:
        print(f"\nMarginal priors: {len(priors['marginal'])} frequencies")

    if 'stratified' in priors:
        n_strata = sum(
            len(sex_priors)
            for age_group in priors['stratified'].values()
            for sex_priors in age_group.values()
        )
        print(f"Stratified priors: {len(priors['stratified'])} age groups × 2 sexes")

    print(f"\nPriors saved to: {builder.output_dir}")

    # Load and display statistics
    stats_path = builder.output_dir / "threshold_statistics.csv"
    if stats_path.exists():
        import pandas as pd
        stats = pd.read_csv(stats_path)
        print("\n--- Sample Statistics (1000 Hz, Right Ear) ---")
        sample = stats[(stats['frequency'] == 1000) & (stats['ear'] == 'right')]
        print(sample[['age_group', 'sex', 'n', 'mean', 'std']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(
        description="Build NHANES-derived priors for Bayesian audiometry"
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download NHANES data (skip if already downloaded)'
    )
    parser.add_argument(
        '--years',
        nargs='+',
        default=RECOMMENDED_CYCLES,
        help='NHANES year cycles to use'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=project_root / 'data' / 'nhanes',
        help='Output directory for data and priors'
    )

    args = parser.parse_args()

    print("="*60)
    print("NHANES PRIOR BUILDER")
    print("="*60)
    print(f"Years: {args.years}")
    print(f"Output directory: {args.output_dir}")

    # Download data if requested
    if args.download:
        download_nhanes_data(args.output_dir, args.years)

    # Build priors
    build_priors(args.output_dir)

    print("\n✅ Prior construction complete!")


if __name__ == '__main__':
    main()
