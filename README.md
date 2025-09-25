# Bayesian Optimization of Pure-Tone Audiometry Through In-Silico Modelling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
<!--[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)-->

## Overview

This repository contains a comprehensive computational framework for simulating and optimizing pure-tone audiometry procedures through large-scale in-silico experiments. We implement both conventional modified Hughson-Westlake (mHW) procedures and novel Bayesian inference methods to estimate hearing thresholds from realistic psychometric response models.

The framework supports research-grade simulations from small validation studies (10-100 listeners) to massive population-scale experiments (10,000+ listeners) with advanced visualization and statistical analysis capabilities.

## Key Features

- **🎧 Realistic Psychometric Modeling**: Sophisticated simulation of human hearing responses including false positives/negatives, attention variability, and diverse hearing loss profiles
- **🔬 Multiple PTA Procedures**:
  - BSA-compliant modified Hughson-Westlake (2018 guidelines)
  - Bayesian adaptive testing with active learning and uncertainty quantification
- **📊 Research-Grade Analysis**:
  - Primary hypothesis testing (H1: Efficiency, H2: Reliability)
  - Test-retest reliability analysis with Bland-Altman plots
  - Statistical power analysis and effect size calculations
- **📈 Advanced Visualization System**:
  - Standard mode: Detailed plots for moderate studies (≤200 listeners)
  - Large-scale mode: Aggregate visualizations for massive studies (1000+ listeners)
  - Publication-ready figures (300 DPI) with clinical formatting
- **⚡ High-Performance Computing**:
  - Parallel processing with configurable worker pools
  - Memory-efficient data handling for large datasets
  - Scalable from 2 to 10,000+ simulated listeners
- **🔬 Reproducible Research**: Full simulation pipeline with comprehensive seed control and parameter tracking

## Installation

### Prerequisites

- Python 3.8 or higher
- Conda/Mamba (recommended) or pip
- Git

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/bayes-audiometry-silico.git
cd bayes-audiometry-silico


2. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate bayes_pta
```

3. Install the package in development mode:
```bash
pip install -e .
```

4. Verify installation:
```bash
python -c "import audiometry_ai; print(audiometry_ai.__version__)"
```

## Usage

### Basic Simulation Example

```python
from audiometry_ai.simulation import SimulatedListener, AudiogramGenerator
from audiometry_ai.procedures import ModifiedHughsonWestlake, BayesianPTA
from audiometry_ai.analysis import compare_procedures

# Create a simulated listener with known thresholds
listener = SimulatedListener(
    true_thresholds={250: 20, 500: 25, 1000: 30, 2000: 35, 4000: 45, 8000: 50},
    false_positive_rate=0.05,
    false_negative_rate=0.05,
    slope=1.5
)

# Run different procedures
mhw = ModifiedHughsonWestlake()
bayes = BayesianPTA(prior='uniform')

mhw_results = mhw.run(listener)
bayes_results = bayes.run(listener)

# Compare results
comparison = compare_procedures(mhw_results, bayes_results, listener.true_thresholds)
print(comparison.summary())
```

### Comprehensive Simulation Testing

The framework includes extensive testing infrastructure in the `tests/` directory:

#### Quick Validation Tests

```bash
# Basic functionality test (2 listeners, ~30 seconds)
cd tests/simple_sim
python simple_sim.py

# Framework validation test (2 listeners with full pipeline)
cd tests/full_mhw_bayes_sim
python full_mhw_bayes_sim.py --config configs/mini_test.yaml --viz standard
```

#### Research-Grade Simulation Studies

```bash
cd tests/full_mhw_bayes_sim

# Small study with detailed visualizations (6 listeners, ~3 minutes)
python full_mhw_bayes_sim.py --config configs/test_config.yaml --viz standard

# Full research study (200 listeners, ~15 minutes)
python full_mhw_bayes_sim.py --config configs/simulation_config.yaml --viz standard

# Large-scale study with optimized visualizations (1000+ listeners)
python full_mhw_bayes_sim.py --config configs/simulation_config.yaml --viz large_scale

# Maximum performance mode (no visualizations for 10,000+ listeners)
python full_mhw_bayes_sim.py --config configs/simulation_config.yaml --viz none
```

#### Visualization Modes

- **`--viz standard`** (default): Detailed plots including individual listener comparisons
- **`--viz large_scale`**: Population-level heatmaps and statistical summaries
- **`--viz none`**: No visualizations for maximum performance

#### Configuration Files

- **`mini_test.yaml`**: Ultra-minimal test (2 listeners, 1 repeat)
- **`test_config.yaml`**: Quick validation (6 listeners, 2 repeats)
- **`simulation_config.yaml`**: Full research study (200 listeners, 2 repeats)

## Project Structure

```
in-silico-audiometry/
├── audiometry_ai/           # Main package
│   ├── simulation/         # Listener models and response generation
│   │   ├── hearing_level_gen.py
│   │   └── response_model.py
│   ├── procedures/         # PTA testing procedures
│   │   ├── bsa_mhw.py     # BSA-compliant modified Hughson-Westlake
│   │   ├── basic_bayes.py # Bayesian adaptive testing
│   │   └── expert_mhw.py  # Expert-level procedures
│   ├── analysis/          # Analysis and metrics calculation
│   │   ├── hearing_level_estimation.py
│   │   └── hearing_level_est_mHW.py
│   ├── visualization/     # Plotting and visualization tools
│   │   ├── bayes_plots.py           # Bayesian-specific visualizations
│   │   ├── hearing_level_visuals.py # Audiogram plotting functions
│   │   └── simulation_plotting.py   # General simulation plots
│   └── utils/             # Utility functions and constants
│       └── defaults.py
├── examples/             # Example scripts organized by type
│   ├── basic/           # Basic mHW examples
│   ├── bayes/          # Bayesian procedure examples
│   └── expert/         # Expert procedure examples
├── tests/               # Comprehensive testing framework
│   ├── simple_sim/     # Basic functionality validation
│   │   ├── simple_sim.py
│   │   ├── results/    # Test output data
│   │   └── viz/        # Test visualizations
│   ├── full_mhw_bayes_sim/  # Research-grade simulation framework
│   │   ├── full_mhw_bayes_sim.py  # Main simulation engine (1500+ lines)
│   │   ├── configs/    # Simulation configuration files
│   │   │   ├── mini_test.yaml        # Ultra-minimal test
│   │   │   ├── test_config.yaml      # Quick validation
│   │   │   └── simulation_config.yaml # Full research study
│   │   ├── results/    # Simulation results with ground truth
│   │   └── viz/        # Publication-quality visualizations
│   └── tests_overview.md    # Detailed testing documentation
├── configs/             # Configuration files
│   └── default.yaml
├── notebooks/          # Jupyter notebooks for analysis
├── environment.yml     # Conda environment
├── package-list.txt   # Pip package requirements
├── setup.py          # Package setup
├── CLAUDE.md        # Claude Code guidance
└── README.md       # This file
```

## Configuration

Simulations are configured via YAML files in `tests/full_mhw_bayes_sim/configs/`. Each configuration defines the complete experimental setup:

### Full Research Configuration (`simulation_config.yaml`)

```yaml
simulation:
  name: "Full mHW vs Bayesian Comparison Study"
  n_listeners: 200          # Sample size for statistical power
  n_repeats: 2             # Test-retest design for reliability
  random_seed: 42          # Reproducibility
  parallel_jobs: 8         # Parallel processing

# Diverse listener population simulation
listeners:
  profiles:
    normal_hearing:         # 25% of population
      proportion: 0.25
      threshold_range: [-10, 20]
      shape: "flat"
    age_related:           # 35% of population
      proportion: 0.35
      threshold_range: [10, 120]
      shape: "high_freq_sloping"
    noise_induced:         # 25% of population
      proportion: 0.25
      threshold_range: [15, 120]
      shape: "4khz_notch"
    mixed_loss:           # 15% of population
      proportion: 0.15
      threshold_range: [20, 120]
      shape: "conductive_sensorineural"

  psychometric:
    slope_range: [3, 15]         # Psychometric curve steepness
    false_positive_range: [0.01, 0.15]
    false_negative_range: [0.01, 0.15]

# Primary outcome measures per manuscript
outcome_measures:
  efficiency: [trial_count, test_duration, convergence_trials]
  accuracy: [absolute_error, bias_analysis, frequency_accuracy]
  reliability: [test_retest_diff, repeatability_coeff, intraclass_correlation]

procedures:
  mhw:
    name: "Modified Hughson-Westlake (BSA 2018)"
    starting_level: 40
    step_size_down: 10
    step_size_up: 5
    max_reversals: 10
    max_trials_per_freq: 20

  bayesian:
    name: "Bayesian Pure-Tone Audiometry"
    prior_type: "uniform"
    prior_range: [-10, 120]
    convergence_threshold: 5.0
    max_trials_per_freq: 30
    active_learning: true
```

## Results and Output

Each simulation generates comprehensive results:

### Data Files
- **`analysis_summary_TIMESTAMP.json`**: Primary hypothesis results and statistical tests
- **`detailed_results_TIMESTAMP.json`**: Complete per-listener data with ground truth parameters

### Visualizations (depends on `--viz` mode)
- **Standard Mode**: Individual listener comparisons, detailed error analysis
- **Large Scale Mode**: Population heatmaps, statistical summaries, efficiency distributions

### Key Metrics Tracked
- **Efficiency**: Trial counts, test duration, convergence analysis
- **Accuracy**: Absolute errors, bias analysis, frequency-specific performance
- **Reliability**: Test-retest differences, Bland-Altman analysis, intraclass correlations
- **Ground Truth**: Complete listener models (thresholds, psychometric parameters, profile types)

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contact

- Liam Barrett - l.barrett.16@ucl.ac.uk

UCL Ear Institute, University College London
