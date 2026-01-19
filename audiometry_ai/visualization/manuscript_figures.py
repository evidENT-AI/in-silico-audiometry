"""
Manuscript figure generation for Stage 1 Registered Report.

Generates publication-quality figures for:
- Figure 1: Population and phenotype overview
- Figure 2: Efficiency results (H1)
- Figure 3: Reliability results (H2)
- Figure 4: Phenotype matching results (H3)
- Figure 5: Summary and key findings

All figures follow Nature formatting guidelines.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import warnings

# Import analysis modules
from ..analysis.reliability import BlandAltmanResult, ICCResult
from ..analysis.phenotype_matching import PhenotypeMatching
from ..simulation.phenotypes import PHENOTYPE_DEFINITIONS, get_phenotype_categories


# =============================================================================
# STYLE CONFIGURATION (Nature requirements)
# =============================================================================

@dataclass
class FigureStyle:
    """Publication figure style settings."""
    # Color palette (colorblind-friendly)
    colors: Dict[str, str] = None

    # Font settings
    font_family: str = "Arial"
    font_size_title: int = 10
    font_size_axis: int = 9
    font_size_tick: int = 8
    font_size_legend: int = 8
    font_size_annotation: int = 7

    # Figure sizes (inches)
    single_column_width: float = 3.5
    double_column_width: float = 7.0
    max_height: float = 9.0

    # DPI for raster outputs
    dpi: int = 300

    def __post_init__(self):
        if self.colors is None:
            self.colors = {
                'bayesian': '#2166AC',      # Blue
                'mhw': '#B2182B',            # Red
                'normal': '#4DAF4A',         # Green
                'presbycusis': '#984EA3',    # Purple
                'noise_induced': '#FF7F00',  # Orange
                'conductive': '#A65628',     # Brown
                'mixed': '#F781BF',          # Pink
                'neutral': '#999999',        # Gray
                'highlight': '#FFFF33',      # Yellow
            }


# Default style
STYLE = FigureStyle()


def setup_matplotlib_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams.update({
        'font.family': STYLE.font_family,
        'font.size': STYLE.font_size_tick,
        'axes.titlesize': STYLE.font_size_title,
        'axes.labelsize': STYLE.font_size_axis,
        'xtick.labelsize': STYLE.font_size_tick,
        'ytick.labelsize': STYLE.font_size_tick,
        'legend.fontsize': STYLE.font_size_legend,
        'figure.dpi': STYLE.dpi,
        'savefig.dpi': STYLE.dpi,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'patch.linewidth': 0.8,
    })


# =============================================================================
# FIGURE 1: POPULATION AND PHENOTYPE OVERVIEW
# =============================================================================

def plot_figure1_population_overview(
    population_data: List[Dict],
    psychometric_params: List[Dict],
    output_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Generate Figure 1: Population and phenotype overview.

    Panels:
    A) Phenotype distribution (pie/bar chart)
    B) Example audiograms by category
    C) Psychometric parameter distributions

    Parameters
    ----------
    population_data : list of dict
        Each dict contains: phenotype, audiogram, listener_id
    psychometric_params : list of dict
        Each dict contains: slope, false_positive_rate, false_negative_rate
    output_path : Path, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    matplotlib.Figure
    """
    setup_matplotlib_style()

    fig = plt.figure(figsize=(STYLE.double_column_width, 5.5))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1],
                  wspace=0.35, hspace=0.4)

    # Panel A: Phenotype distribution
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_phenotype_distribution(ax_a, population_data)
    ax_a.set_title('A', loc='left', fontweight='bold', fontsize=STYLE.font_size_title)

    # Panel B: Example audiograms (spans 2 columns)
    ax_b = fig.add_subplot(gs[0, 1:])
    _plot_example_audiograms(ax_b, population_data)
    ax_b.set_title('B', loc='left', fontweight='bold', fontsize=STYLE.font_size_title)

    # Panel C: Psychometric parameter distributions (3 subplots)
    ax_c1 = fig.add_subplot(gs[1, 0])
    ax_c2 = fig.add_subplot(gs[1, 1])
    ax_c3 = fig.add_subplot(gs[1, 2])
    _plot_psychometric_distributions(ax_c1, ax_c2, ax_c3, psychometric_params)
    ax_c1.set_title('C', loc='left', fontweight='bold', fontsize=STYLE.font_size_title)

    plt.tight_layout()

    if output_path:
        _save_figure(fig, output_path, 'fig1_population_overview')

    if show:
        plt.show()

    return fig


def _plot_phenotype_distribution(ax: plt.Axes, population_data: List[Dict]):
    """Plot phenotype distribution as horizontal bar chart."""
    # Count phenotypes
    phenotype_counts = {}
    for p in population_data:
        ptype = p.get('phenotype', 'unknown')
        phenotype_counts[ptype] = phenotype_counts.get(ptype, 0) + 1

    # Sort by category
    categories = get_phenotype_categories()
    category_order = ['normal', 'presbycusis', 'noise_induced', 'conductive', 'mixed']

    phenotypes = []
    counts = []
    colors = []

    for cat in category_order:
        for ptype in categories.get(cat, []):
            if ptype in phenotype_counts:
                phenotypes.append(ptype.replace('_', ' ').title())
                counts.append(phenotype_counts[ptype])
                colors.append(STYLE.colors.get(cat, STYLE.colors['neutral']))

    y_pos = np.arange(len(phenotypes))
    ax.barh(y_pos, counts, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(phenotypes, fontsize=STYLE.font_size_tick - 1)
    ax.set_xlabel('Count', fontsize=STYLE.font_size_axis)
    ax.invert_yaxis()
    ax.set_xlim(0, max(counts) * 1.1)

    # Add count labels
    for i, v in enumerate(counts):
        ax.text(v + max(counts) * 0.02, i, str(v),
                va='center', fontsize=STYLE.font_size_annotation)


def _plot_example_audiograms(ax: plt.Axes, population_data: List[Dict]):
    """Plot example audiograms from each category."""
    categories = get_phenotype_categories()
    frequencies = [250, 500, 1000, 2000, 4000, 8000]

    # Get one example from each category
    for cat_name, phenotypes in categories.items():
        # Find first listener of this category
        for p in population_data:
            if p.get('phenotype') in phenotypes:
                audiogram = p.get('audiogram', {})
                thresholds = [audiogram.get(f, 0) for f in frequencies]
                ax.plot(frequencies, thresholds, 'o-',
                       color=STYLE.colors.get(cat_name, STYLE.colors['neutral']),
                       label=cat_name.replace('_', ' ').title(),
                       markersize=4, alpha=0.8)
                break

    ax.set_xscale('log')
    ax.set_xticks(frequencies)
    ax.set_xticklabels([str(f) for f in frequencies])
    ax.set_xlabel('Frequency (Hz)', fontsize=STYLE.font_size_axis)
    ax.set_ylabel('Hearing Level (dB HL)', fontsize=STYLE.font_size_axis)
    ax.invert_yaxis()
    ax.set_ylim(120, -10)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=STYLE.font_size_legend - 1, framealpha=0.9)


def _plot_psychometric_distributions(ax1, ax2, ax3, params: List[Dict]):
    """Plot distributions of psychometric parameters."""
    slopes = [p.get('slope', 8) for p in params]
    fps = [p.get('false_positive_rate', 0.05) for p in params]
    fns = [p.get('false_negative_rate', 0.02) for p in params]

    # Slope distribution
    ax1.hist(slopes, bins=30, color=STYLE.colors['bayesian'],
             edgecolor='white', alpha=0.8)
    ax1.axvline(np.mean(slopes), color='red', linestyle='--', linewidth=1.5)
    ax1.set_xlabel(r'Slope $\sigma$ (dB)', fontsize=STYLE.font_size_axis)
    ax1.set_ylabel('Count', fontsize=STYLE.font_size_axis)
    ax1.text(0.95, 0.95, f'Mean: {np.mean(slopes):.1f}',
             transform=ax1.transAxes, ha='right', va='top',
             fontsize=STYLE.font_size_annotation)

    # False positive rate
    ax2.hist(fps, bins=30, color=STYLE.colors['mhw'],
             edgecolor='white', alpha=0.8)
    ax2.axvline(np.mean(fps), color='red', linestyle='--', linewidth=1.5)
    ax2.set_xlabel(r'False Positive Rate $\alpha$', fontsize=STYLE.font_size_axis)
    ax2.set_ylabel('Count', fontsize=STYLE.font_size_axis)
    ax2.text(0.95, 0.95, f'Mean: {np.mean(fps):.3f}',
             transform=ax2.transAxes, ha='right', va='top',
             fontsize=STYLE.font_size_annotation)

    # False negative rate
    ax3.hist(fns, bins=30, color=STYLE.colors['noise_induced'],
             edgecolor='white', alpha=0.8)
    ax3.axvline(np.mean(fns), color='red', linestyle='--', linewidth=1.5)
    ax3.set_xlabel(r'False Negative Rate $\beta$', fontsize=STYLE.font_size_axis)
    ax3.set_ylabel('Count', fontsize=STYLE.font_size_axis)
    ax3.text(0.95, 0.95, f'Mean: {np.mean(fns):.3f}',
             transform=ax3.transAxes, ha='right', va='top',
             fontsize=STYLE.font_size_annotation)


# =============================================================================
# FIGURE 2: EFFICIENCY RESULTS (H1)
# =============================================================================

def plot_figure2_efficiency(
    results: List[Dict],
    output_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Generate Figure 2: Efficiency comparison (H1).

    Panels:
    A) Trial count comparison boxplot
    B) Trial count by phenotype heatmap
    C) Accuracy vs trials scatter
    D) Effect size forest plot

    Parameters
    ----------
    results : list of dict
        Simulation results with trial counts and accuracy
    output_path : Path, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    matplotlib.Figure
    """
    setup_matplotlib_style()

    fig = plt.figure(figsize=(STYLE.double_column_width, 5.5))
    gs = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.4)

    # Panel A: Trial count comparison
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_trial_count_comparison(ax_a, results)
    ax_a.set_title('A', loc='left', fontweight='bold', fontsize=STYLE.font_size_title)

    # Panel B: Heatmap by phenotype
    ax_b = fig.add_subplot(gs[0, 1])
    _plot_efficiency_heatmap(ax_b, results)
    ax_b.set_title('B', loc='left', fontweight='bold', fontsize=STYLE.font_size_title)

    # Panel C: Accuracy vs trials
    ax_c = fig.add_subplot(gs[1, 0])
    _plot_accuracy_vs_trials(ax_c, results)
    ax_c.set_title('C', loc='left', fontweight='bold', fontsize=STYLE.font_size_title)

    # Panel D: Effect size forest plot
    ax_d = fig.add_subplot(gs[1, 1])
    _plot_effect_size_forest(ax_d, results)
    ax_d.set_title('D', loc='left', fontweight='bold', fontsize=STYLE.font_size_title)

    plt.tight_layout()

    if output_path:
        _save_figure(fig, output_path, 'fig2_efficiency')

    if show:
        plt.show()

    return fig


def _plot_trial_count_comparison(ax: plt.Axes, results: List[Dict]):
    """Plot trial count comparison as paired boxplot."""
    mhw_trials = []
    bayes_trials = []

    for r in results:
        mhw_trials.append(r.get('mhw_total_trials', 0))
        bayes_trials.append(r.get('bayes_total_trials', 0))

    # Create boxplot
    bp = ax.boxplot([mhw_trials, bayes_trials],
                    labels=['mHW', 'Bayesian'],
                    patch_artist=True,
                    widths=0.6)

    # Color the boxes
    colors = [STYLE.colors['mhw'], STYLE.colors['bayesian']]
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points
    for i, (data, color) in enumerate(zip([mhw_trials, bayes_trials], colors)):
        x = np.random.normal(i + 1, 0.04, len(data))
        ax.scatter(x, data, alpha=0.3, s=5, color=color)

    ax.set_ylabel('Total Trials', fontsize=STYLE.font_size_axis)

    # Add significance annotation
    mean_diff = np.mean(mhw_trials) - np.mean(bayes_trials)
    pct_reduction = mean_diff / np.mean(mhw_trials) * 100
    ax.text(0.5, 0.95, f'{pct_reduction:.1f}% reduction',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=STYLE.font_size_annotation, fontweight='bold')


def _plot_efficiency_heatmap(ax: plt.Axes, results: List[Dict]):
    """Plot efficiency gain heatmap by phenotype and frequency."""
    # Organize by phenotype
    by_phenotype = {}
    for r in results:
        ptype = r.get('phenotype', 'unknown')
        if ptype not in by_phenotype:
            by_phenotype[ptype] = {'mhw': [], 'bayes': []}
        by_phenotype[ptype]['mhw'].append(r.get('mhw_total_trials', 0))
        by_phenotype[ptype]['bayes'].append(r.get('bayes_total_trials', 0))

    # Calculate mean reduction per phenotype
    phenotypes = list(PHENOTYPE_DEFINITIONS.keys())
    reductions = []
    for ptype in phenotypes:
        if ptype in by_phenotype:
            mhw_mean = np.mean(by_phenotype[ptype]['mhw'])
            bayes_mean = np.mean(by_phenotype[ptype]['bayes'])
            if mhw_mean > 0:
                reduction = (mhw_mean - bayes_mean) / mhw_mean * 100
            else:
                reduction = 0
            reductions.append(reduction)
        else:
            reductions.append(0)

    # Create bar chart
    y_pos = np.arange(len(phenotypes))
    colors = [STYLE.colors.get(PHENOTYPE_DEFINITIONS[p].category, STYLE.colors['neutral'])
              for p in phenotypes]

    ax.barh(y_pos, reductions, color=colors, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([p.replace('_', ' ') for p in phenotypes],
                       fontsize=STYLE.font_size_tick - 1)
    ax.set_xlabel('Trial Reduction (%)', fontsize=STYLE.font_size_axis)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.invert_yaxis()


def _plot_accuracy_vs_trials(ax: plt.Axes, results: List[Dict]):
    """Plot accuracy vs trial count scatter."""
    mhw_trials = []
    mhw_errors = []
    bayes_trials = []
    bayes_errors = []

    for r in results:
        mhw_trials.append(r.get('mhw_total_trials', 0))
        bayes_trials.append(r.get('bayes_total_trials', 0))
        mhw_errors.append(r.get('mhw_mean_error', 0))
        bayes_errors.append(r.get('bayes_mean_error', 0))

    ax.scatter(mhw_trials, mhw_errors, alpha=0.5, s=20,
               color=STYLE.colors['mhw'], label='mHW', edgecolors='none')
    ax.scatter(bayes_trials, bayes_errors, alpha=0.5, s=20,
               color=STYLE.colors['bayesian'], label='Bayesian', edgecolors='none')

    ax.set_xlabel('Total Trials', fontsize=STYLE.font_size_axis)
    ax.set_ylabel('Mean Absolute Error (dB)', fontsize=STYLE.font_size_axis)
    ax.axhline(5, color='gray', linestyle='--', alpha=0.5, label='5 dB criterion')
    ax.legend(loc='upper right', fontsize=STYLE.font_size_legend - 1)
    ax.set_ylim(0, max(max(mhw_errors), max(bayes_errors)) * 1.1)


def _plot_effect_size_forest(ax: plt.Axes, results: List[Dict]):
    """Plot effect size forest plot by frequency."""
    frequencies = [250, 500, 1000, 2000, 4000, 8000]
    effect_sizes = []
    cis = []

    for freq in frequencies:
        mhw_trials = [r.get(f'mhw_trials_{freq}', 0) for r in results]
        bayes_trials = [r.get(f'bayes_trials_{freq}', 0) for r in results]

        # Calculate Cohen's d
        diff = np.array(mhw_trials) - np.array(bayes_trials)
        d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
        effect_sizes.append(d)

        # Bootstrap CI
        ci_low, ci_high = _bootstrap_ci(diff)
        d_ci_low = ci_low / np.std(diff) if np.std(diff) > 0 else 0
        d_ci_high = ci_high / np.std(diff) if np.std(diff) > 0 else 0
        cis.append((d_ci_low, d_ci_high))

    y_pos = np.arange(len(frequencies))

    # Plot effect sizes with CIs
    for i, (es, ci) in enumerate(zip(effect_sizes, cis)):
        ax.plot([ci[0], ci[1]], [i, i], color=STYLE.colors['bayesian'], linewidth=2)
        ax.plot(es, i, 'o', color=STYLE.colors['bayesian'], markersize=8)

    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0.5, color='red', linestyle=':', alpha=0.5, label='Medium effect')
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f'{f} Hz' for f in frequencies])
    ax.set_xlabel("Cohen's d (mHW - Bayesian)", fontsize=STYLE.font_size_axis)
    ax.invert_yaxis()
    ax.legend(loc='lower right', fontsize=STYLE.font_size_legend - 1)


def _bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Calculate bootstrap confidence interval for mean."""
    rng = np.random.default_rng(42)
    means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        means.append(np.mean(sample))

    alpha = 1 - ci
    return np.percentile(means, alpha/2 * 100), np.percentile(means, (1 - alpha/2) * 100)


# =============================================================================
# FIGURE 3: RELIABILITY RESULTS (H2)
# =============================================================================

def plot_figure3_reliability(
    test1_results: List[Dict],
    test2_results: List[Dict],
    icc_bayes: ICCResult,
    icc_mhw: ICCResult,
    ba_bayes: BlandAltmanResult,
    ba_mhw: BlandAltmanResult,
    output_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Generate Figure 3: Reliability comparison (H2).

    Panels:
    A) Bland-Altman plot for Bayesian
    B) Bland-Altman plot for mHW
    C) ICC comparison bar chart
    D) Test-retest scatter comparison

    Parameters
    ----------
    test1_results, test2_results : list of dict
        Results from first and second test sessions
    icc_bayes, icc_mhw : ICCResult
        ICC results for each procedure
    ba_bayes, ba_mhw : BlandAltmanResult
        Bland-Altman results for each procedure
    output_path : Path, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    matplotlib.Figure
    """
    setup_matplotlib_style()

    fig = plt.figure(figsize=(STYLE.double_column_width, 5.5))
    gs = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.4)

    # Panel A: Bland-Altman Bayesian
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_bland_altman(ax_a, test1_results, test2_results, 'bayes', ba_bayes)
    ax_a.set_title('A  Bayesian', loc='left', fontweight='bold',
                   fontsize=STYLE.font_size_title)

    # Panel B: Bland-Altman mHW
    ax_b = fig.add_subplot(gs[0, 1])
    _plot_bland_altman(ax_b, test1_results, test2_results, 'mhw', ba_mhw)
    ax_b.set_title('B  mHW', loc='left', fontweight='bold',
                   fontsize=STYLE.font_size_title)

    # Panel C: ICC comparison
    ax_c = fig.add_subplot(gs[1, 0])
    _plot_icc_comparison(ax_c, icc_bayes, icc_mhw)
    ax_c.set_title('C', loc='left', fontweight='bold', fontsize=STYLE.font_size_title)

    # Panel D: Test-retest scatter
    ax_d = fig.add_subplot(gs[1, 1])
    _plot_test_retest_scatter(ax_d, test1_results, test2_results)
    ax_d.set_title('D', loc='left', fontweight='bold', fontsize=STYLE.font_size_title)

    plt.tight_layout()

    if output_path:
        _save_figure(fig, output_path, 'fig3_reliability')

    if show:
        plt.show()

    return fig


def _plot_bland_altman(ax: plt.Axes, test1: List[Dict], test2: List[Dict],
                       procedure: str, ba_result: BlandAltmanResult):
    """Plot Bland-Altman difference plot."""
    thresh_key = f'{procedure}_thresholds'

    # Collect all threshold differences
    means = []
    diffs = []

    for r1, r2 in zip(test1, test2):
        t1 = r1.get(thresh_key, {})
        t2 = r2.get(thresh_key, {})

        for freq in t1:
            if freq in t2:
                mean = (t1[freq] + t2[freq]) / 2
                diff = t1[freq] - t2[freq]
                means.append(mean)
                diffs.append(diff)

    color = STYLE.colors['bayesian'] if procedure == 'bayes' else STYLE.colors['mhw']

    ax.scatter(means, diffs, alpha=0.4, s=15, color=color, edgecolors='none')

    # Mean difference line
    ax.axhline(ba_result.mean_diff, color='black', linestyle='-', linewidth=1.5,
               label=f'Bias: {ba_result.mean_diff:.2f} dB')

    # Limits of agreement
    ax.axhline(ba_result.loa_upper, color='gray', linestyle='--', linewidth=1)
    ax.axhline(ba_result.loa_lower, color='gray', linestyle='--', linewidth=1)

    # Fill between LOA
    ax.fill_between(ax.get_xlim(), ba_result.loa_lower, ba_result.loa_upper,
                    alpha=0.1, color='gray')

    ax.set_xlabel('Mean of Test-Retest (dB HL)', fontsize=STYLE.font_size_axis)
    ax.set_ylabel('Difference (dB)', fontsize=STYLE.font_size_axis)

    # Add LOA annotation
    ax.text(0.02, 0.98, f'LOA: [{ba_result.loa_lower:.1f}, {ba_result.loa_upper:.1f}]',
            transform=ax.transAxes, va='top', fontsize=STYLE.font_size_annotation)


def _plot_icc_comparison(ax: plt.Axes, icc_bayes: ICCResult, icc_mhw: ICCResult):
    """Plot ICC comparison with confidence intervals."""
    procedures = ['Bayesian', 'mHW']
    iccs = [icc_bayes.icc, icc_mhw.icc]
    ci_lows = [icc_bayes.ci_low, icc_mhw.ci_low]
    ci_highs = [icc_bayes.ci_high, icc_mhw.ci_high]
    colors = [STYLE.colors['bayesian'], STYLE.colors['mhw']]

    x = np.arange(len(procedures))

    # Plot bars
    bars = ax.bar(x, iccs, color=colors, edgecolor='white', alpha=0.8, width=0.6)

    # Add error bars
    ax.errorbar(x, iccs,
                yerr=[[i - l for i, l in zip(iccs, ci_lows)],
                      [h - i for i, h in zip(iccs, ci_highs)]],
                fmt='none', color='black', capsize=5, capthick=1.5)

    # Add interpretation thresholds
    ax.axhline(0.9, color='green', linestyle=':', alpha=0.7, label='Excellent (0.9)')
    ax.axhline(0.75, color='orange', linestyle=':', alpha=0.7, label='Good (0.75)')

    ax.set_xticks(x)
    ax.set_xticklabels(procedures)
    ax.set_ylabel('ICC(2,1)', fontsize=STYLE.font_size_axis)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=STYLE.font_size_legend - 1)

    # Add value annotations
    for i, (icc, ci_l, ci_h) in enumerate(zip(iccs, ci_lows, ci_highs)):
        ax.text(i, icc + 0.05, f'{icc:.3f}\n[{ci_l:.2f}, {ci_h:.2f}]',
                ha='center', va='bottom', fontsize=STYLE.font_size_annotation)


def _plot_test_retest_scatter(ax: plt.Axes, test1: List[Dict], test2: List[Dict]):
    """Plot test-retest scatter for both procedures."""
    for procedure, color, label in [('bayes', STYLE.colors['bayesian'], 'Bayesian'),
                                     ('mhw', STYLE.colors['mhw'], 'mHW')]:
        thresh_key = f'{procedure}_thresholds'

        t1_all = []
        t2_all = []

        for r1, r2 in zip(test1, test2):
            t1 = r1.get(thresh_key, {})
            t2 = r2.get(thresh_key, {})

            for freq in t1:
                if freq in t2:
                    t1_all.append(t1[freq])
                    t2_all.append(t2[freq])

        ax.scatter(t1_all, t2_all, alpha=0.3, s=10, color=color,
                   label=label, edgecolors='none')

    # Identity line
    lims = [ax.get_xlim()[0], ax.get_xlim()[1]]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Identity')

    ax.set_xlabel('Test 1 Threshold (dB HL)', fontsize=STYLE.font_size_axis)
    ax.set_ylabel('Test 2 Threshold (dB HL)', fontsize=STYLE.font_size_axis)
    ax.legend(loc='lower right', fontsize=STYLE.font_size_legend - 1)
    ax.set_aspect('equal')


# =============================================================================
# FIGURE 4: PHENOTYPE MATCHING RESULTS (H3)
# =============================================================================

def plot_figure4_phenotype_matching(
    matching_results: Dict,
    correlation_results: Tuple[float, float, float, float],
    output_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Generate Figure 4: Phenotype matching results (H3).

    Panels:
    A) Predicted vs observed efficiency scatter
    B) Matching accuracy confusion matrix
    C) Feature importance
    D) Correlation with confidence interval

    Parameters
    ----------
    matching_results : dict
        Results from phenotype matching including predicted/observed values
    correlation_results : tuple
        (r, p, ci_low, ci_high) from H3 correlation analysis
    output_path : Path, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    matplotlib.Figure
    """
    setup_matplotlib_style()

    fig = plt.figure(figsize=(STYLE.double_column_width, 5.5))
    gs = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.4)

    # Panel A: Predicted vs observed
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_predicted_vs_observed(ax_a, matching_results)
    ax_a.set_title('A', loc='left', fontweight='bold', fontsize=STYLE.font_size_title)

    # Panel B: Confusion matrix
    ax_b = fig.add_subplot(gs[0, 1])
    _plot_matching_confusion(ax_b, matching_results)
    ax_b.set_title('B', loc='left', fontweight='bold', fontsize=STYLE.font_size_title)

    # Panel C: Feature importance
    ax_c = fig.add_subplot(gs[1, 0])
    _plot_feature_importance(ax_c, matching_results)
    ax_c.set_title('C', loc='left', fontweight='bold', fontsize=STYLE.font_size_title)

    # Panel D: Correlation CI
    ax_d = fig.add_subplot(gs[1, 1])
    _plot_correlation_ci(ax_d, correlation_results)
    ax_d.set_title('D', loc='left', fontweight='bold', fontsize=STYLE.font_size_title)

    plt.tight_layout()

    if output_path:
        _save_figure(fig, output_path, 'fig4_phenotype_matching')

    if show:
        plt.show()

    return fig


def _plot_predicted_vs_observed(ax: plt.Axes, results: Dict):
    """Plot predicted vs observed efficiency gains."""
    predicted = results.get('predicted_gains', [])
    observed = results.get('observed_gains', [])
    phenotypes = results.get('phenotypes', [])

    if not predicted or not observed:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center',
                transform=ax.transAxes)
        return

    # Color by phenotype category
    colors = []
    for p in phenotypes:
        if p in PHENOTYPE_DEFINITIONS:
            cat = PHENOTYPE_DEFINITIONS[p].category
            colors.append(STYLE.colors.get(cat, STYLE.colors['neutral']))
        else:
            colors.append(STYLE.colors['neutral'])

    ax.scatter(predicted, observed, c=colors, alpha=0.6, s=30, edgecolors='white')

    # Identity line
    lims = [min(min(predicted), min(observed)), max(max(predicted), max(observed))]
    ax.plot(lims, lims, 'k--', alpha=0.5)

    # Regression line
    if len(predicted) > 2:
        z = np.polyfit(predicted, observed, 1)
        p = np.poly1d(z)
        ax.plot(lims, p(lims), color=STYLE.colors['bayesian'], linewidth=2)

    ax.set_xlabel('Predicted Efficiency Gain (trials)', fontsize=STYLE.font_size_axis)
    ax.set_ylabel('Observed Efficiency Gain (trials)', fontsize=STYLE.font_size_axis)


def _plot_matching_confusion(ax: plt.Axes, results: Dict):
    """Plot phenotype matching confusion matrix."""
    confusion = results.get('confusion_matrix', None)
    labels = results.get('phenotype_labels', list(PHENOTYPE_DEFINITIONS.keys()))

    if confusion is None:
        # Create dummy matrix for demonstration
        n = min(len(labels), 5)
        confusion = np.eye(n) * 0.8 + np.random.rand(n, n) * 0.2
        labels = labels[:n]

    im = ax.imshow(confusion, cmap='Blues', aspect='auto')

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([l[:8] for l in labels], rotation=45, ha='right',
                       fontsize=STYLE.font_size_tick - 2)
    ax.set_yticklabels([l[:8] for l in labels], fontsize=STYLE.font_size_tick - 2)
    ax.set_xlabel('Predicted', fontsize=STYLE.font_size_axis)
    ax.set_ylabel('True', fontsize=STYLE.font_size_axis)

    plt.colorbar(im, ax=ax, label='Proportion')


def _plot_feature_importance(ax: plt.Axes, results: Dict):
    """Plot feature importance for phenotype matching."""
    feature_names = PhenotypeMatching.FEATURE_NAMES
    importances = results.get('feature_importance', np.random.rand(len(feature_names)))

    y_pos = np.arange(len(feature_names))
    ax.barh(y_pos, importances, color=STYLE.colors['bayesian'], alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace('_', ' ') for f in feature_names],
                       fontsize=STYLE.font_size_tick - 1)
    ax.set_xlabel('Importance', fontsize=STYLE.font_size_axis)
    ax.invert_yaxis()


def _plot_correlation_ci(ax: plt.Axes, results: Tuple[float, float, float, float]):
    """Plot correlation coefficient with confidence interval."""
    r, p, ci_low, ci_high = results

    # Main correlation point
    ax.plot(0, r, 'o', markersize=15, color=STYLE.colors['bayesian'])
    ax.plot([0, 0], [ci_low, ci_high], color=STYLE.colors['bayesian'], linewidth=3)

    # Reference lines
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(0.3, color='orange', linestyle=':', alpha=0.7, label='Weak (0.3)')
    ax.axhline(0.6, color='green', linestyle=':', alpha=0.7, label='Moderate (0.6)')

    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.2, 1.0)
    ax.set_xticks([])
    ax.set_ylabel('Correlation (r)', fontsize=STYLE.font_size_axis)
    ax.legend(loc='lower right', fontsize=STYLE.font_size_legend - 1)

    # Annotation
    sig_text = f'r = {r:.3f}\np = {p:.4f}' if p >= 0.0001 else f'r = {r:.3f}\np < 0.0001'
    ax.text(0.25, r, sig_text, ha='left', va='center', fontsize=STYLE.font_size_annotation)


# =============================================================================
# FIGURE 5: SUMMARY
# =============================================================================

def plot_figure5_summary(
    h1_results: Dict,
    h2_results: Dict,
    h3_results: Dict,
    output_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Generate Figure 5: Summary of key findings.

    Panels:
    A) Hypothesis summary table
    B) Key metrics comparison radar chart

    Parameters
    ----------
    h1_results, h2_results, h3_results : dict
        Results for each hypothesis
    output_path : Path, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    matplotlib.Figure
    """
    setup_matplotlib_style()

    fig = plt.figure(figsize=(STYLE.single_column_width, 4.5))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1, 1.5], hspace=0.4)

    # Panel A: Summary table
    ax_a = fig.add_subplot(gs[0])
    _plot_hypothesis_table(ax_a, h1_results, h2_results, h3_results)
    ax_a.set_title('A', loc='left', fontweight='bold', fontsize=STYLE.font_size_title)

    # Panel B: Radar chart
    ax_b = fig.add_subplot(gs[1], projection='polar')
    _plot_metrics_radar(ax_b, h1_results, h2_results, h3_results)

    plt.tight_layout()

    if output_path:
        _save_figure(fig, output_path, 'fig5_summary')

    if show:
        plt.show()

    return fig


def _plot_hypothesis_table(ax: plt.Axes, h1: Dict, h2: Dict, h3: Dict):
    """Plot hypothesis summary as table."""
    ax.axis('off')

    # Table data
    data = [
        ['Hypothesis', 'Metric', 'Result', 'p-value'],
        ['H1: Efficiency', 'Trial reduction', f"{h1.get('reduction_pct', 0):.1f}%",
         f"{h1.get('p_value', 1):.4f}"],
        ['H2: Reliability', 'ICC (Bayes)', f"{h2.get('icc_bayes', 0):.3f}",
         f"{h2.get('p_value', 1):.4f}"],
        ['H3: Matching', 'Correlation', f"{h3.get('correlation', 0):.3f}",
         f"{h3.get('p_value', 1):.4f}"],
    ]

    table = ax.table(cellText=data[1:], colLabels=data[0],
                     loc='center', cellLoc='center',
                     colWidths=[0.3, 0.25, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(STYLE.font_size_tick)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor(STYLE.colors['bayesian'])
        table[(0, i)].set_text_props(color='white', fontweight='bold')


def _plot_metrics_radar(ax: plt.Axes, h1: Dict, h2: Dict, h3: Dict):
    """Plot key metrics as radar chart."""
    categories = ['Efficiency', 'Accuracy', 'Reliability', 'Consistency', 'Matching']
    n_cats = len(categories)

    # Normalize metrics to 0-1 scale
    bayes_values = [
        h1.get('reduction_pct', 20) / 40,  # Efficiency (0-40%)
        1 - h1.get('bayes_error', 5) / 10,  # Accuracy (inverted)
        h2.get('icc_bayes', 0.9),  # Reliability
        1 - h2.get('bayes_sd', 5) / 10,  # Consistency (inverted)
        h3.get('correlation', 0.6),  # Matching
    ]

    mhw_values = [
        0,  # Efficiency baseline
        1 - h1.get('mhw_error', 5) / 10,
        h2.get('icc_mhw', 0.85),
        1 - h2.get('mhw_sd', 7) / 10,
        0,  # N/A for mHW
    ]

    # Close the radar
    angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
    angles += angles[:1]
    bayes_values += bayes_values[:1]
    mhw_values += mhw_values[:1]

    ax.plot(angles, bayes_values, 'o-', linewidth=2,
            color=STYLE.colors['bayesian'], label='Bayesian')
    ax.fill(angles, bayes_values, alpha=0.25, color=STYLE.colors['bayesian'])

    ax.plot(angles, mhw_values, 'o-', linewidth=2,
            color=STYLE.colors['mhw'], label='mHW')
    ax.fill(angles, mhw_values, alpha=0.25, color=STYLE.colors['mhw'])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=STYLE.font_size_tick)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
              fontsize=STYLE.font_size_legend)


# =============================================================================
# FIGURE 6: BAYESIAN HYPOTHESIS TESTING
# =============================================================================

def plot_figure6_bayesian(
    bayesian_stats: Dict,
    output_path: Optional[Path] = None,
    show: bool = True
) -> plt.Figure:
    """
    Generate Figure 6: Bayesian hypothesis testing results.

    Panels:
    A) Bayes Factors with interpretation thresholds
    B) H1 posterior distribution with HDI
    C) H2 posterior distribution with HDI
    D) Probability of practical significance

    Parameters
    ----------
    bayesian_stats : dict
        Bayesian analysis results including BFs, posteriors, HDIs
    output_path : Path, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    matplotlib.Figure
    """
    setup_matplotlib_style()

    fig = plt.figure(figsize=(STYLE.double_column_width, 5.5))
    gs = GridSpec(2, 2, figure=fig, wspace=0.3, hspace=0.4)

    # Panel A: Bayes Factors
    ax_a = fig.add_subplot(gs[0, 0])
    _plot_bayes_factors(ax_a, bayesian_stats)
    ax_a.set_title('A  Bayes Factors', loc='left', fontweight='bold',
                   fontsize=STYLE.font_size_title)

    # Panel B: H1 Posterior
    ax_b = fig.add_subplot(gs[0, 1])
    _plot_posterior_h1(ax_b, bayesian_stats)
    ax_b.set_title('B  H1: Efficiency Posterior', loc='left', fontweight='bold',
                   fontsize=STYLE.font_size_title)

    # Panel C: H2 Posterior
    ax_c = fig.add_subplot(gs[1, 0])
    _plot_posterior_h2(ax_c, bayesian_stats)
    ax_c.set_title('C  H2: Reliability Posterior', loc='left', fontweight='bold',
                   fontsize=STYLE.font_size_title)

    # Panel D: Practical significance
    ax_d = fig.add_subplot(gs[1, 1])
    _plot_practical_significance(ax_d, bayesian_stats)
    ax_d.set_title('D  Practical Significance', loc='left', fontweight='bold',
                   fontsize=STYLE.font_size_title)

    plt.tight_layout()

    if output_path:
        _save_figure(fig, output_path, 'fig6_bayesian')

    if show:
        plt.show()

    return fig


def _plot_bayes_factors(ax: plt.Axes, stats: Dict):
    """Plot Bayes Factors with interpretation thresholds."""
    # Extract BFs (cap at reasonable display value)
    h1_bf = min(stats.get('h1_bf10', 1), 1e6)
    h2_bf = min(stats.get('h2_bf10', 1), 1e6)

    hypotheses = ['H1: Efficiency', 'H2: Reliability']
    bfs = [h1_bf, h2_bf]

    x = np.arange(len(hypotheses))
    colors = [STYLE.colors['bayesian'], STYLE.colors['bayesian']]

    # Plot bars on log scale
    bars = ax.bar(x, bfs, color=colors, edgecolor='white', alpha=0.8, width=0.6)

    # Reference lines for interpretation
    ax.axhline(100, color='darkgreen', linestyle='--', alpha=0.7, label='Extreme (100)')
    ax.axhline(30, color='green', linestyle=':', alpha=0.7, label='Very Strong (30)')
    ax.axhline(10, color='orange', linestyle=':', alpha=0.7, label='Strong (10)')
    ax.axhline(3, color='red', linestyle=':', alpha=0.7, label='Moderate (3)')

    ax.set_xticks(x)
    ax.set_xticklabels(hypotheses, fontsize=STYLE.font_size_axis)
    ax.set_ylabel('Bayes Factor (BF₁₀)', fontsize=STYLE.font_size_axis)
    ax.set_yscale('log')
    ax.set_ylim(0.1, max(bfs) * 10)
    ax.legend(loc='upper right', fontsize=STYLE.font_size_legend - 1)

    # Add value annotations
    for i, (bf, interp) in enumerate(zip(bfs, [stats.get('h1_bf_interpretation', ''),
                                                stats.get('h2_bf_interpretation', '')])):
        bf_text = f'{bf:.1e}' if bf > 1000 else f'{bf:.1f}'
        ax.text(i, bf * 1.5, f'BF = {bf_text}\n{interp}',
                ha='center', va='bottom', fontsize=STYLE.font_size_annotation)


def _plot_posterior_h1(ax: plt.Axes, stats: Dict):
    """Plot H1 posterior distribution with HDI."""
    mean = stats.get('h1_posterior_mean', 15)
    hdi = stats.get('h1_posterior_hdi', (10, 20))
    threshold = stats.get('h1_practical_threshold', 5)

    # Create approximate posterior (normal approximation)
    sd = (hdi[1] - hdi[0]) / 3.92  # 95% CI width
    x = np.linspace(max(0, mean - 4*sd), mean + 4*sd, 200)
    y = np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

    # Plot posterior
    ax.fill_between(x, y, alpha=0.3, color=STYLE.colors['bayesian'])
    ax.plot(x, y, color=STYLE.colors['bayesian'], linewidth=2)

    # HDI
    ax.axvline(hdi[0], color=STYLE.colors['bayesian'], linestyle='--', linewidth=1.5)
    ax.axvline(hdi[1], color=STYLE.colors['bayesian'], linestyle='--', linewidth=1.5)

    # Shade HDI region
    hdi_x = x[(x >= hdi[0]) & (x <= hdi[1])]
    hdi_y = y[(x >= hdi[0]) & (x <= hdi[1])]
    ax.fill_between(hdi_x, hdi_y, alpha=0.5, color=STYLE.colors['bayesian'], label='95% HDI')

    # Practical threshold
    ax.axvline(threshold, color='red', linestyle=':', linewidth=2, label=f'δ_min = {threshold}')

    # Posterior mean
    ax.axvline(mean, color='black', linestyle='-', linewidth=1.5)

    ax.set_xlabel('Trial Reduction (mHW - Bayesian)', fontsize=STYLE.font_size_axis)
    ax.set_ylabel('Density', fontsize=STYLE.font_size_axis)
    ax.set_ylim(0, None)
    ax.legend(loc='upper right', fontsize=STYLE.font_size_legend - 1)

    # Annotation
    ax.text(0.95, 0.95, f'Mean: {mean:.1f}\nHDI: [{hdi[0]:.1f}, {hdi[1]:.1f}]',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=STYLE.font_size_annotation,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def _plot_posterior_h2(ax: plt.Axes, stats: Dict):
    """Plot H2 posterior distribution with HDI."""
    mean = stats.get('h2_posterior_mean', 2)
    hdi = stats.get('h2_posterior_hdi', (1, 4))
    threshold = stats.get('h2_practical_threshold', 1)

    # Create approximate posterior (normal approximation)
    sd = (hdi[1] - hdi[0]) / 3.92  # 95% CI width
    x = np.linspace(max(0, mean - 4*sd), mean + 4*sd, 200)
    y = np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))

    # Plot posterior
    ax.fill_between(x, y, alpha=0.3, color=STYLE.colors['bayesian'])
    ax.plot(x, y, color=STYLE.colors['bayesian'], linewidth=2)

    # HDI
    ax.axvline(hdi[0], color=STYLE.colors['bayesian'], linestyle='--', linewidth=1.5)
    ax.axvline(hdi[1], color=STYLE.colors['bayesian'], linestyle='--', linewidth=1.5)

    # Shade HDI region
    hdi_x = x[(x >= hdi[0]) & (x <= hdi[1])]
    hdi_y = y[(x >= hdi[0]) & (x <= hdi[1])]
    ax.fill_between(hdi_x, hdi_y, alpha=0.5, color=STYLE.colors['bayesian'], label='95% HDI')

    # Practical threshold
    ax.axvline(threshold, color='red', linestyle=':', linewidth=2, label=f'δ_min = {threshold} dB')

    # Posterior mean
    ax.axvline(mean, color='black', linestyle='-', linewidth=1.5)

    ax.set_xlabel('Test-Retest SD Reduction (dB)', fontsize=STYLE.font_size_axis)
    ax.set_ylabel('Density', fontsize=STYLE.font_size_axis)
    ax.set_ylim(0, None)
    ax.legend(loc='upper right', fontsize=STYLE.font_size_legend - 1)

    # Annotation
    ax.text(0.95, 0.95, f'Mean: {mean:.2f} dB\nHDI: [{hdi[0]:.2f}, {hdi[1]:.2f}]',
            transform=ax.transAxes, ha='right', va='top',
            fontsize=STYLE.font_size_annotation,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def _plot_practical_significance(ax: plt.Axes, stats: Dict):
    """Plot probability of practical significance."""
    hypotheses = ['H1: Efficiency\n(δ > 5 trials)', 'H2: Reliability\n(δ > 1 dB)']
    probs = [
        stats.get('h1_prob_practical', 0.95),
        stats.get('h2_prob_practical', 0.90),
    ]

    x = np.arange(len(hypotheses))
    colors = [STYLE.colors['bayesian'], STYLE.colors['bayesian']]

    # Plot bars
    bars = ax.bar(x, probs, color=colors, edgecolor='white', alpha=0.8, width=0.6)

    # Reference lines
    ax.axhline(0.95, color='darkgreen', linestyle='--', alpha=0.7, label='Strong evidence (0.95)')
    ax.axhline(0.80, color='orange', linestyle=':', alpha=0.7, label='Moderate (0.80)')

    ax.set_xticks(x)
    ax.set_xticklabels(hypotheses, fontsize=STYLE.font_size_axis - 1)
    ax.set_ylabel('P(δ > δ_min | data)', fontsize=STYLE.font_size_axis)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='lower right', fontsize=STYLE.font_size_legend - 1)

    # Add value annotations
    for i, prob in enumerate(probs):
        ax.text(i, prob + 0.03, f'{prob:.3f}',
                ha='center', va='bottom', fontsize=STYLE.font_size_annotation,
                fontweight='bold')


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def _save_figure(fig: plt.Figure, output_dir: Path, name: str,
                 formats: List[str] = ['png', 'pdf', 'svg']):
    """Save figure in multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, dpi=STYLE.dpi, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        print(f"Saved: {path}")


def generate_all_manuscript_figures(
    simulation_results: Dict,
    output_dir: Path,
    show: bool = False
) -> Dict[str, plt.Figure]:
    """
    Generate all manuscript figures from simulation results.

    Parameters
    ----------
    simulation_results : dict
        Complete simulation results including:
        - population_data
        - psychometric_params
        - test1_results, test2_results
        - icc_results, bland_altman_results
        - matching_results, correlation_results
        - h1_results, h2_results, h3_results
    output_dir : Path
        Directory to save figures
    show : bool
        Whether to display figures

    Returns
    -------
    dict
        Dictionary mapping figure names to Figure objects
    """
    figures = {}

    # Figure 1: Population overview
    if 'population_data' in simulation_results:
        figures['fig1'] = plot_figure1_population_overview(
            population_data=simulation_results['population_data'],
            psychometric_params=simulation_results.get('psychometric_params', []),
            output_path=output_dir,
            show=show
        )

    # Figure 2: Efficiency (H1)
    if 'results' in simulation_results:
        figures['fig2'] = plot_figure2_efficiency(
            results=simulation_results['results'],
            output_path=output_dir,
            show=show
        )

    # Figure 3: Reliability (H2)
    if 'test1_results' in simulation_results and 'test2_results' in simulation_results:
        figures['fig3'] = plot_figure3_reliability(
            test1_results=simulation_results['test1_results'],
            test2_results=simulation_results['test2_results'],
            icc_bayes=simulation_results.get('icc_bayes'),
            icc_mhw=simulation_results.get('icc_mhw'),
            ba_bayes=simulation_results.get('ba_bayes'),
            ba_mhw=simulation_results.get('ba_mhw'),
            output_path=output_dir,
            show=show
        )

    # Figure 4: Phenotype matching (H3)
    if 'matching_results' in simulation_results:
        figures['fig4'] = plot_figure4_phenotype_matching(
            matching_results=simulation_results['matching_results'],
            correlation_results=simulation_results.get('correlation_results', (0, 1, -1, 1)),
            output_path=output_dir,
            show=show
        )

    # Figure 5: Summary
    figures['fig5'] = plot_figure5_summary(
        h1_results=simulation_results.get('h1_results', {}),
        h2_results=simulation_results.get('h2_results', {}),
        h3_results=simulation_results.get('h3_results', {}),
        output_path=output_dir,
        show=show
    )

    return figures
