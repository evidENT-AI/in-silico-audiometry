"""
Visualization module for audiometry results and analysis.

This module contains functions for:
- Plotting audiograms and hearing profiles
- Visualizing Bayesian inference results
- Creating comparison plots between procedures
- Publication-quality manuscript figures (Stage 1)
"""

from .bayes_plots import *
from .hearing_level_visuals import *
from .simulation_plotting import *
from .manuscript_figures import (
    plot_figure1_population_overview,
    plot_figure2_efficiency,
    plot_figure3_reliability,
    plot_figure4_phenotype_matching,
    plot_figure5_summary,
    generate_all_manuscript_figures,
    setup_matplotlib_style,
    FigureStyle,
    STYLE,
)

__all__ = [
    # Manuscript figures
    'plot_figure1_population_overview',
    'plot_figure2_efficiency',
    'plot_figure3_reliability',
    'plot_figure4_phenotype_matching',
    'plot_figure5_summary',
    'generate_all_manuscript_figures',
    'setup_matplotlib_style',
    'FigureStyle',
    'STYLE',
]