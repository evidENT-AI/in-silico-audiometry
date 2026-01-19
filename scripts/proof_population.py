#!/usr/bin/env python3
"""
Population Proofing Script

Preview simulated patient audiograms before running full simulation.
Allows adjustment of phenotype distribution and visual verification.

Usage:
    python scripts/proof_population.py
    python scripts/proof_population.py --preset normal_majority
    python scripts/proof_population.py --output proofs/
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import sys
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from audiometry_ai.simulation.phenotypes import (
    PhenotypeGenerator,
    PHENOTYPE_DEFINITIONS,
    FREQUENCIES,
    get_phenotype_categories,
)

# Import DemographicGenerator from run_stage1_simulation
from run_stage1_simulation import DemographicGenerator


# =============================================================================
# DISTRIBUTION PRESETS
# =============================================================================

DISTRIBUTION_PRESETS = {
    # Original RNENT clinical distribution
    'rnent_clinical': {
        'moderate_sloping': 0.11,
        'mild_sloping': 0.08,
        'severe_profound': 0.03,
        'near_normal_mild_hf': 0.12,
        'moderate_high_freq': 0.08,
        'moderate_severe': 0.10,
        'mild_hf_drop': 0.20,
        'ski_slope': 0.15,
        'normal_hearing': 0.13,
    },

    # Normal hearing majority (for healthy population screening)
    'normal_majority': {
        'moderate_sloping': 0.05,
        'mild_sloping': 0.05,
        'severe_profound': 0.02,
        'near_normal_mild_hf': 0.10,
        'moderate_high_freq': 0.05,
        'moderate_severe': 0.03,
        'mild_hf_drop': 0.10,
        'ski_slope': 0.05,
        'normal_hearing': 0.55,  # 55% normal hearing
    },

    # Balanced for method comparison
    'balanced': {
        'moderate_sloping': 0.11,
        'mild_sloping': 0.11,
        'severe_profound': 0.11,
        'near_normal_mild_hf': 0.11,
        'moderate_high_freq': 0.11,
        'moderate_severe': 0.11,
        'mild_hf_drop': 0.11,
        'ski_slope': 0.11,
        'normal_hearing': 0.12,
    },

    # Hearing loss focused (clinical population)
    'clinical_hl': {
        'moderate_sloping': 0.15,
        'mild_sloping': 0.10,
        'severe_profound': 0.05,
        'near_normal_mild_hf': 0.10,
        'moderate_high_freq': 0.10,
        'moderate_severe': 0.10,
        'mild_hf_drop': 0.15,
        'ski_slope': 0.15,
        'normal_hearing': 0.10,
    },
}


def distribution_to_counts(distribution: dict, n_total: int = 2200) -> dict:
    """Convert proportions to counts for n_total listeners."""
    # Initial allocation
    counts = {k: int(v * n_total) for k, v in distribution.items()}

    # Adjust to match exact total
    current = sum(counts.values())
    diff = n_total - current

    if diff != 0:
        # Add/remove from largest category
        largest = max(counts, key=counts.get)
        counts[largest] += diff

    return counts


def generate_preview_population(
    distribution: dict,
    n_preview: int = 100,
    seed: int = 42
) -> tuple:
    """
    Generate a preview population for proofing.

    Returns
    -------
    tuple
        (population_data, stats_by_phenotype, demographics_summary)
    """
    phenotype_gen = PhenotypeGenerator()
    demo_gen = DemographicGenerator(seed=seed)

    # Scale distribution to preview size
    counts = distribution_to_counts(distribution, n_preview)

    # Ensure at least 1 of each phenotype for preview
    for k in counts:
        if counts[k] < 1:
            counts[k] = 1

    # Regenerate to match preview size
    total = sum(counts.values())
    if total != n_preview:
        scale = n_preview / total
        counts = {k: max(1, int(v * scale)) for k, v in counts.items()}
        # Final adjustment
        diff = n_preview - sum(counts.values())
        if diff != 0:
            largest = max(counts, key=counts.get)
            counts[largest] += diff

    # Generate population
    population = phenotype_gen.generate_population(counts, seed=seed)

    # Generate demographics for each listener
    demographics_list = [
        demo_gen.generate(p['phenotype'], p['category'])
        for p in population
    ]

    # Compute demographics summary
    ages = [d['age'] for d in demographics_list]
    demographics_summary = {
        'age_mean': np.mean(ages),
        'age_std': np.std(ages),
        'age_min': min(ages),
        'age_max': max(ages),
        'male_pct': sum(1 for d in demographics_list if d['sex'] == 'male') / len(demographics_list) * 100,
        'female_pct': sum(1 for d in demographics_list if d['sex'] == 'female') / len(demographics_list) * 100,
        # Risk factors
        'diabetes_pct': sum(1 for d in demographics_list if d['covariates'].get('diabetes', False)) / len(demographics_list) * 100,
        'cv_risk_pct': sum(1 for d in demographics_list if d['covariates'].get('cardiovascular_risk', False)) / len(demographics_list) * 100,
        'noise_exposure_pct': sum(1 for d in demographics_list if d['covariates'].get('noise_exposure', False)) / len(demographics_list) * 100,
        'tinnitus_pct': sum(1 for d in demographics_list if d['covariates'].get('tinnitus', False)) / len(demographics_list) * 100,
        'ototoxic_pct': sum(1 for d in demographics_list if d['covariates'].get('ototoxic_medication', False)) / len(demographics_list) * 100,
        'menieres_pct': sum(1 for d in demographics_list if d['covariates'].get('menieres', False)) / len(demographics_list) * 100,
    }

    # Compute statistics by phenotype
    stats = {}
    for phenotype in PHENOTYPE_DEFINITIONS.keys():
        listeners = [p for p in population if p['phenotype'] == phenotype]
        if listeners:
            audiograms = np.array([
                [p['audiogram'][f] for f in FREQUENCIES]
                for p in listeners
            ])
            stats[phenotype] = {
                'n': len(listeners),
                'mean': np.mean(audiograms, axis=0),
                'std': np.std(audiograms, axis=0),
                'min': np.min(audiograms, axis=0),
                'max': np.max(audiograms, axis=0),
                'audiograms': audiograms,
            }
        else:
            stats[phenotype] = {
                'n': 0,
                'mean': np.zeros(len(FREQUENCIES)),
                'std': np.zeros(len(FREQUENCIES)),
                'min': np.zeros(len(FREQUENCIES)),
                'max': np.zeros(len(FREQUENCIES)),
                'audiograms': np.array([]),
            }

    return population, stats, demographics_summary


def plot_population_proof(
    stats: dict,
    distribution: dict,
    n_total: int,
    preset_name: str,
    output_path: Path = None,
    show: bool = True
) -> plt.Figure:
    """
    Create proof visualization of the population.

    Shows:
    - Distribution pie chart
    - Mean audiograms by phenotype
    - Audiogram spread (min/max bands)
    - Summary statistics table
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Color scheme
    category_colors = {
        'normal': '#4DAF4A',
        'mild': '#377EB8',
        'presbycusis': '#984EA3',
        'noise_induced': '#FF7F00',
        'severe': '#E41A1C',
    }

    phenotype_colors = {
        name: category_colors.get(defn.category, '#999999')
        for name, defn in PHENOTYPE_DEFINITIONS.items()
    }

    # Panel A: Distribution pie chart
    ax_pie = fig.add_subplot(gs[0, 0])
    labels = []
    sizes = []
    colors = []
    for name, prop in distribution.items():
        if prop > 0:
            labels.append(name.replace('_', '\n'))
            sizes.append(prop)
            colors.append(phenotype_colors[name])

    wedges, texts, autotexts = ax_pie.pie(
        sizes, labels=None, colors=colors, autopct='%1.0f%%',
        startangle=90, pctdistance=0.75
    )
    ax_pie.set_title(f'A  Distribution: {preset_name}\n(n={n_total})',
                     fontweight='bold', fontsize=10)

    # Add legend below pie
    ax_pie.legend(wedges, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  ncol=3, fontsize=7)

    # Panel B: Mean audiograms by phenotype
    ax_audio = fig.add_subplot(gs[0, 1:])
    for name, s in stats.items():
        if s['n'] > 0:
            ax_audio.plot(FREQUENCIES, s['mean'], 'o-',
                         color=phenotype_colors[name],
                         label=f"{name} (n={s['n']})",
                         markersize=4, alpha=0.8)

    ax_audio.set_xscale('log')
    ax_audio.set_xticks(FREQUENCIES)
    ax_audio.set_xticklabels([str(f) for f in FREQUENCIES])
    ax_audio.set_xlabel('Frequency (Hz)')
    ax_audio.set_ylabel('Hearing Level (dB HL)')
    ax_audio.invert_yaxis()
    ax_audio.set_ylim(120, -10)
    ax_audio.grid(True, alpha=0.3)
    ax_audio.legend(loc='lower right', fontsize=7, ncol=2)
    ax_audio.set_title('B  Mean Audiograms by Phenotype', fontweight='bold', fontsize=10)

    # Panels C-E: Individual phenotype details (top 3 by count)
    sorted_phenotypes = sorted(stats.items(), key=lambda x: x[1]['n'], reverse=True)

    for idx, (name, s) in enumerate(sorted_phenotypes[:3]):
        ax = fig.add_subplot(gs[1, idx])
        if s['n'] > 0 and len(s['audiograms']) > 0:
            # Plot individual audiograms (light)
            for audiogram in s['audiograms'][:20]:  # Max 20 for clarity
                ax.plot(FREQUENCIES, audiogram, '-',
                       color=phenotype_colors[name], alpha=0.2, linewidth=0.5)

            # Plot mean with error bands
            ax.fill_between(FREQUENCIES,
                           s['mean'] - s['std'],
                           s['mean'] + s['std'],
                           color=phenotype_colors[name], alpha=0.3)
            ax.plot(FREQUENCIES, s['mean'], 'o-',
                   color=phenotype_colors[name], linewidth=2, markersize=5)

        ax.set_xscale('log')
        ax.set_xticks(FREQUENCIES)
        ax.set_xticklabels([str(f) for f in FREQUENCIES], fontsize=7)
        ax.set_xlabel('Frequency (Hz)', fontsize=8)
        if idx == 0:
            ax.set_ylabel('Hearing Level (dB HL)', fontsize=8)
        ax.invert_yaxis()
        ax.set_ylim(120, -10)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{chr(67+idx)}  {name.replace("_", " ").title()}\n(n={s["n"]}, {distribution[name]*100:.0f}%)',
                    fontweight='bold', fontsize=9)

    # Panel F: Summary statistics table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')

    # Create table data
    table_data = [['Phenotype', 'Category', 'n', '%', 'Mean PTA', 'SD', 'Range']]
    for name in PHENOTYPE_DEFINITIONS.keys():
        s = stats[name]
        defn = PHENOTYPE_DEFINITIONS[name]
        if s['n'] > 0:
            # Pure-tone average (500, 1000, 2000 Hz)
            pta_idx = [FREQUENCIES.index(f) for f in [500, 1000, 2000]]
            pta_mean = np.mean(s['mean'][pta_idx])
            pta_std = np.mean(s['std'][pta_idx])
            pta_range = f"{np.min(s['min']):.0f}-{np.max(s['max']):.0f}"
        else:
            pta_mean = 0
            pta_std = 0
            pta_range = "-"

        table_data.append([
            name.replace('_', ' '),
            defn.category,
            str(s['n']),
            f"{distribution[name]*100:.1f}%",
            f"{pta_mean:.1f}",
            f"{pta_std:.1f}",
            pta_range
        ])

    table = ax_table.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc='center',
        cellLoc='center',
        colWidths=[0.18, 0.12, 0.08, 0.08, 0.12, 0.10, 0.12]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.4)

    # Color header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#2166AC')
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Color rows by category
    for row_idx, name in enumerate(PHENOTYPE_DEFINITIONS.keys(), start=1):
        color = phenotype_colors[name]
        for col_idx in range(len(table_data[0])):
            table[(row_idx, col_idx)].set_facecolor(color + '30')  # 30% opacity

    ax_table.set_title('F  Population Summary Statistics', fontweight='bold',
                       fontsize=10, pad=20)

    plt.suptitle(f'Population Proof: {preset_name.upper()} Distribution\n'
                 f'Total: {n_total} listeners | Preview: {sum(s["n"] for s in stats.values())} samples',
                 fontsize=12, fontweight='bold', y=0.98)

    if output_path:
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        for fmt in ['png', 'pdf']:
            fig.savefig(output_path / f'population_proof_{preset_name}.{fmt}',
                       dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved proof to: {output_path}/population_proof_{preset_name}.{{png,pdf}}")

    if show:
        plt.show()

    return fig


def print_distribution_summary(distribution: dict, n_total: int, preset_name: str):
    """Print summary of distribution to console."""
    counts = distribution_to_counts(distribution, n_total)

    print(f"\n{'='*60}")
    print(f"POPULATION DISTRIBUTION: {preset_name.upper()}")
    print(f"{'='*60}")
    print(f"Total listeners: {n_total}")
    print()

    # Group by category
    categories = get_phenotype_categories()
    category_totals = {}

    print(f"{'Phenotype':<25} {'Category':<15} {'Count':>8} {'%':>8}")
    print("-" * 60)

    for name in PHENOTYPE_DEFINITIONS.keys():
        defn = PHENOTYPE_DEFINITIONS[name]
        count = counts[name]
        pct = distribution[name] * 100
        print(f"{name:<25} {defn.category:<15} {count:>8} {pct:>7.1f}%")

        if defn.category not in category_totals:
            category_totals[defn.category] = 0
        category_totals[defn.category] += count

    print("-" * 60)
    print("\nBy Category:")
    for cat, total in sorted(category_totals.items(), key=lambda x: -x[1]):
        pct = total / n_total * 100
        print(f"  {cat:<20} {total:>8} ({pct:.1f}%)")

    print()


def print_demographics_summary(demographics_summary: dict):
    """Print demographics and risk factor summary."""
    print(f"\n{'='*60}")
    print("DEMOGRAPHICS & RISK FACTORS (NHANES-based)")
    print(f"{'='*60}")
    print(f"\nAge Distribution:")
    print(f"  Mean: {demographics_summary['age_mean']:.1f} years")
    print(f"  SD:   {demographics_summary['age_std']:.1f} years")
    print(f"  Range: {demographics_summary['age_min']}-{demographics_summary['age_max']} years")

    print(f"\nSex Distribution:")
    print(f"  Male:   {demographics_summary['male_pct']:.1f}%")
    print(f"  Female: {demographics_summary['female_pct']:.1f}%")

    print(f"\nRisk Factors (Level 2 Prior Conditioning):")
    print(f"  Diabetes:           {demographics_summary['diabetes_pct']:.1f}%")
    print(f"  Cardiovascular:     {demographics_summary['cv_risk_pct']:.1f}%")
    print(f"  Noise Exposure:     {demographics_summary['noise_exposure_pct']:.1f}%")
    print(f"  Tinnitus:           {demographics_summary['tinnitus_pct']:.1f}%")
    print(f"  Ototoxic Meds:      {demographics_summary['ototoxic_pct']:.1f}%")
    print(f"  Meniere's Disease:  {demographics_summary['menieres_pct']:.1f}%")
    print()


def save_distribution_config(distribution: dict, output_path: Path, preset_name: str):
    """Save distribution config to JSON for reproducibility."""
    config = {
        'preset_name': preset_name,
        'distribution': distribution,
        'n_total': 2200,
        'counts': distribution_to_counts(distribution, 2200),
    }
    config_path = output_path / f'distribution_config_{preset_name}.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preview simulated patient population before full simulation"
    )
    parser.add_argument(
        '--preset', '-p', type=str, default='normal_majority',
        choices=list(DISTRIBUTION_PRESETS.keys()),
        help='Distribution preset to use'
    )
    parser.add_argument(
        '--n_preview', type=int, default=200,
        help='Number of samples for preview (default: 200)'
    )
    parser.add_argument(
        '--n_total', type=int, default=2200,
        help='Total listeners for full simulation (default: 2200)'
    )
    parser.add_argument(
        '--output', '-o', type=str, default='results/proofs',
        help='Output directory for proof figures'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--no-show', action='store_true',
        help='Do not display figures interactively'
    )
    parser.add_argument(
        '--list-presets', action='store_true',
        help='List available distribution presets'
    )

    args = parser.parse_args()

    if args.list_presets:
        print("\nAvailable distribution presets:")
        print("-" * 40)
        for name, dist in DISTRIBUTION_PRESETS.items():
            normal_pct = dist.get('normal_hearing', 0) * 100
            print(f"  {name:<20} (normal: {normal_pct:.0f}%)")
        print()
        return

    # Get distribution
    distribution = DISTRIBUTION_PRESETS[args.preset]
    output_path = Path(args.output)

    # Print summary
    print_distribution_summary(distribution, args.n_total, args.preset)

    # Generate preview population
    print(f"Generating preview population (n={args.n_preview})...")
    population, stats, demographics_summary = generate_preview_population(
        distribution, args.n_preview, args.seed
    )

    # Print demographics summary
    print_demographics_summary(demographics_summary)

    # Create proof visualization
    print("Creating proof visualization...")
    plot_population_proof(
        stats=stats,
        distribution=distribution,
        n_total=args.n_total,
        preset_name=args.preset,
        output_path=output_path,
        show=not args.no_show
    )

    # Save config
    save_distribution_config(distribution, output_path, args.preset)

    print(f"\n{'='*60}")
    print("NEXT STEPS")
    print("="*60)
    print("1. Review the proof visualization")
    print("2. If approved, run full simulation with:")
    print(f"   python scripts/run_stage1_simulation.py \\")
    print(f"       --n_listeners {args.n_total} \\")
    print(f"       --distribution {args.preset} \\")
    print(f"       --nhanes")
    print()
    print("   This will use NHANES priors with full three-level conditioning:")
    print("     - Level 1: Age-sex stratification")
    print("     - Level 2: Risk factors (diabetes, CV, noise, etc.)")
    print("     - Level 3: Tympanometry (clinical only)")
    print()


if __name__ == "__main__":
    main()
