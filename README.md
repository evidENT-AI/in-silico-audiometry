# Bayesian Optimization of Pure-Tone Audiometry Through In-Silico Modelling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
<!--[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)-->

## Overview

This repository contains the computational framework for simulating and optimizing pure-tone audiometry procedures through in-silico experiments. We implement both conventional modified Hughson-Westlake (mHW) procedures and novel Bayesian inference methods to estimate hearing thresholds from simulated psychometric responses.

<!--This work is part of a Stage 1 Registered Report submitted to Royal Society Open Science, comparing automated audiometry approaches against conventional testing through both simulated and human participant validation.-->

## Key Features

- **Psychometric Response Simulation**: Realistic modeling of human responses including false positives/negatives
- **Multiple PTA Procedures**: 
  - BSA-compliant modified Hughson-Westlake
  - Bayesian adaptive testing with uncertainty quantification
- **Comprehensive Analysis**: Test-retest reliability, convergence metrics, and efficiency comparisons
- **Reproducible Research**: Full simulation pipeline with seed control for reproducibility

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
conda activate audiometry-ai
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

### Running Full Simulation Study

```bash
# Run main simulation with default parameters
python scripts/run_simulation.py

# Custom configuration
python scripts/run_simulation.py --config configs/custom_sim.yaml

# Parallel execution for large-scale simulations
python scripts/run_simulation.py --n-listeners 1000 --n-jobs 8
```

## Project Structure

```
bayes-audiometry-silico/
├── audiometry_ai/           # Main package
│   ├── simulation/         # Listener models and response generation
│   │   ├── listener.py
│   │   ├── psychometric.py
│   │   └── audiogram.py
│   ├── procedures/         # PTA testing procedures
│   │   ├── base.py
│   │   ├── hughson_westlake.py
│   │   └── bayesian.py
│   ├── analysis/          # Analysis and visualization tools
│   │   ├── metrics.py
│   │   ├── plotting.py
│   │   └── statistics.py
│   └── utils/             # Utility functions
├── configs/               # Configuration files
│   ├── default.yaml
│   └── experiments/
├── data/                  # Data directory
│   ├── raw/              # Raw simulation outputs
│   ├── processed/        # Processed results
│   └── figures/          # Generated figures
├── notebooks/            # Jupyter notebooks
│   ├── 01_psychometric_modeling.ipynb
│   ├── 02_procedure_comparison.ipynb
│   └── 03_results_analysis.ipynb
├── scripts/              # Executable scripts
│   ├── run_simulation.py
│   └── reproduce_*.py
├── tests/                # Unit tests
├── docs/                 # Documentation
├── environment.yml       # Conda environment
├── requirements.txt      # Pip requirements
├── setup.py             # Package setup
└── README.md            # This file
```

## Configuration

Simulations can be configured via YAML files. See `configs/default.yaml` for available options:

```yaml
simulation:
  n_listeners: 100
  n_repeats: 3
  seed: 42
  
listener:
  phenotypes: ['normal', 'age_related', 'noise_induced']
  false_positive_range: [0.01, 0.1]
  false_negative_range: [0.01, 0.1]
  
procedures:
  mhw:
    starting_level: 40
    max_reversals: 10
  bayesian:
    prior: 'uniform'
    convergence_threshold: 5.0
```

## Testing

Run the test suite:

```bash
# All tests
pytest

# With coverage
pytest --cov=audiometry_ai --cov-report=html

# Specific test modules
pytest tests/test_simulation.py
pytest tests/test_procedures.py
```

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Contact

- Liam Barrett - l.barrett.16@ucl.ac.uk

UCL Ear Institute, University College London
