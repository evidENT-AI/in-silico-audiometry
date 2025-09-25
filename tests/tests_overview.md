# Tests Overview

This directory contains test suites for the audiometry simulation framework. Each test is organized in its own subdirectory with a standardized structure for scripts, results, and visualizations.

## Test Structure

Each test folder follows this standardized structure:

```
test_name/
├── test_name.py        # Main test script
├── results/           # Data output directory
│   ├── *.json        # Detailed results in JSON format
│   └── *.csv         # Threshold data in CSV format
└── viz/              # Visualization output directory
    └── *.png         # Generated plots and figures
```

## Available Tests

### `simple_sim/` - Simple Simulation Test

**Purpose**: Basic functionality test for both mHW and Bayesian procedures on a known hearing profile.

**What it tests**:
- Modified Hughson-Westlake (mHW) procedure accuracy
- Bayesian Pure-Tone Audiometry procedure accuracy
- Comparison between methods
- Error analysis and visualization

**Test Profile**:
- Mild to moderate hearing loss (10-50 dB HL across frequencies)
- Frequencies: 250, 500, 1000, 2000, 4000, 8000 Hz
- Known ground truth for accuracy validation

**Outputs**:
- `results/mhw_results_TIMESTAMP.json` - Complete mHW test results
- `results/bayes_results_TIMESTAMP.json` - Complete Bayesian test results
- `results/mhw_thresholds_TIMESTAMP.csv` - mHW threshold estimates in CSV format
- `results/bayes_thresholds_TIMESTAMP.csv` - Bayesian threshold estimates in CSV format
- `viz/threshold_comparison_TIMESTAMP.png` - Side-by-side comparison of all methods
- `viz/error_analysis_TIMESTAMP.png` - Error analysis and accuracy metrics

**Usage**:
```bash
cd tests/simple_sim
python simple_sim.py
```

or from repository root:
```bash
python tests/simple_sim/simple_sim.py
```

**Expected Results**:
- Both procedures should achieve < 3 dB mean absolute error
- Test should complete successfully with data and visualizations generated
- Comparison plots should show close agreement with true thresholds

### `full_mhw_bayes_sim/` - Comprehensive mHW vs Bayesian Comparison Study

**Purpose**: Full-scale simulation study comparing mHW vs Bayesian procedures across diverse listener populations. Based on Stage 1 Registered Report methodology.

**What it tests**:
- **Primary Hypothesis H1**: Efficiency (N_Bayes < N_mHW) - Bayesian requires fewer trials
- **Primary Hypothesis H2**: Reliability - Bayesian shows better test-retest reliability
- Accuracy within ±5 dB HL criterion across diverse listener types
- Performance across different psychometric parameters and hearing loss patterns

**Test Population**:
- **200+ simulated listeners** with diverse hearing profiles:
  - Normal hearing (25%): 0-15 dB HL flat profiles
  - Age-related hearing loss (35%): High-frequency sloping losses
  - Noise-induced hearing loss (25%): 4kHz notch patterns
  - Mixed hearing loss (15%): Combined conductive + sensorineural
- **Psychometric diversity**: Variable slopes (3-15 dB), false positive/negative rates (1-15%)
- **Test-retest design**: Each listener tested twice for reliability assessment

**Outputs**:
- `results/analysis_summary_TIMESTAMP.json` - Primary outcome measures and statistical tests
- `results/detailed_results_TIMESTAMP.json` - Complete per-listener simulation data
- `viz/efficiency_comparison_TIMESTAMP.png` - Trial count comparisons (H1)
- `viz/reliability_analysis_TIMESTAMP.png` - Test-retest reliability plots (H2)
- `viz/accuracy_heatmaps_TIMESTAMP.png` - Performance by frequency and listener type
- `configs/simulation_config.yaml` - Complete parameter specification

**Usage**:
```bash
cd tests/full_mhw_bayes_sim
python full_mhw_bayes_sim.py
```

**Expected Results**:
- **H1 Efficiency**: Bayesian procedure should require significantly fewer trials than mHW
- **H2 Reliability**: Bayesian should show better test-retest reliability (smaller differences)
- **Accuracy**: Both procedures should achieve >90% accuracy within ±5 dB HL
- **Statistical Power**: Sample size designed for adequate power to detect meaningful differences
- **Execution Time**: ~5-15 minutes depending on system (parallel processing enabled)

## Running Tests

### Individual Tests
Navigate to the specific test directory and run the Python script:
```bash
cd tests/simple_sim
python simple_sim.py
```

### All Tests (Future)
When multiple tests exist, a master test runner will be available:
```bash
python run_all_tests.py  # (To be implemented)
```

## Output Files

### Results Data
- **JSON files** contain complete test results including:
  - Timestamp and test parameters
  - True vs estimated thresholds
  - Progression patterns (detailed test sequences)
  - Accuracy metrics and error analysis
  - Success/failure status and any error messages

- **CSV files** contain simplified threshold data for easy import into analysis tools:
  - Frequency (Hz)
  - True threshold (dB HL)
  - Estimated threshold (dB HL)
  - Absolute error (dB)

### Visualizations
- **Comparison plots** show true thresholds vs estimates from each method
- **Error analysis plots** show accuracy metrics and frequency-specific errors
- **Method-specific plots** show detailed results for individual procedures

## Adding New Tests

To add a new test:

1. Create a new directory: `tests/new_test_name/`
2. Create subdirectories: `results/` and `viz/`
3. Create the test script: `new_test_name.py`
4. Follow the established output format (JSON + CSV for data, PNG for visualizations)
5. Update this overview file with test description

## Test Data Management

- Results are timestamped to avoid overwrites
- Large result files should be added to `.gitignore` to keep repository clean
- Visualization files should be reasonably sized (< 5MB per image)
- Consider archiving old test results periodically

## Dependencies

Tests require the full audiometry_ai package and its dependencies:
- numpy, scipy, matplotlib for core functionality
- pandas for data handling (if used)
- All audiometry_ai modules and their dependencies

## Troubleshooting

**Import Errors**: Ensure the audiometry_ai package is installed (`pip install -e .` from root)

**Permission Errors**: Make sure test scripts are executable (`chmod +x test_script.py`)

**Missing Directories**: The test scripts should create `results/` and `viz/` directories automatically

**Plot Display Issues**: Tests save plots to files rather than displaying them to work in headless environments

For additional help, check the main repository documentation or individual test script docstrings.