"""
NHANES-derived prior distributions for Bayesian audiometry.

Builds kernel density estimates (KDEs) from NHANES audiometry data,
stratified by age, sex, and other covariates to create informed priors
for threshold estimation.

Based on manuscript specification:
- Level 1: Age-sex stratification
- Level 2: Risk factor adjustments (diabetes, cardiovascular, noise, tinnitus)
- Level 3: Tympanometric conditioning
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from ..utils.logging import setup_logger
from ..utils.io import ensure_directory_exists
from ..data.constants import (
    PTA_COLUMNS,
    NHANES_FREQUENCIES,
    STANDARD_FREQUENCIES,
    AGE_GROUPS,
    THRESHOLD_BOUNDS,
    DEFAULT_KDE_BANDWIDTH,
)

logger = setup_logger(__name__)


class NHANESPriorBuilder:
    """
    Build hierarchical prior distributions from NHANES audiometry data.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing NHANES CSV files
    output_dir : str or Path, optional
        Directory for saving computed priors. Defaults to data_dir/priors
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / "priors"
        ensure_directory_exists(self.output_dir)

        # Storage for loaded data and computed priors
        self.audiometry_data: Optional[pd.DataFrame] = None
        self.demographics_data: Optional[pd.DataFrame] = None
        self.merged_data: Optional[pd.DataFrame] = None
        self.priors: Dict = {}

        logger.info(f"NHANESPriorBuilder initialized")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Output directory: {self.output_dir}")

    def load_data(
        self,
        audiometry_pattern: str = "**/Audiometry/*.csv",
        demographics_pattern: str = "**/Demographics/*.csv"
    ) -> pd.DataFrame:
        """
        Load and merge NHANES audiometry and demographics data.

        Parameters
        ----------
        audiometry_pattern : str
            Glob pattern for audiometry CSV files
        demographics_pattern : str
            Glob pattern for demographics CSV files

        Returns
        -------
        pd.DataFrame
            Merged dataset with audiometry and demographics
        """
        # Find audiometry files
        audio_files = list(self.data_dir.glob(audiometry_pattern))
        if not audio_files:
            # Try the csv subdirectory structure
            audio_files = list(self.data_dir.glob("csv/Audiometry/*.csv"))

        if not audio_files:
            raise FileNotFoundError(f"No audiometry files found matching {audiometry_pattern}")

        logger.info(f"Found {len(audio_files)} audiometry file(s)")

        # Load and concatenate audiometry data
        audio_dfs = []
        for f in audio_files:
            if 'readable' not in str(f):
                df = pd.read_csv(f, low_memory=False)
                audio_dfs.append(df)
                logger.info(f"Loaded {f.name}: {len(df)} rows")

        self.audiometry_data = pd.concat(audio_dfs, ignore_index=True)
        logger.info(f"Total audiometry records: {len(self.audiometry_data)}")

        # Find demographics files
        demo_files = list(self.data_dir.glob(demographics_pattern))
        if not demo_files:
            demo_files = list(self.data_dir.glob("csv/Demographics/*.csv"))
        if not demo_files:
            # Try NHANES-specific naming convention
            demo_files = list(self.data_dir.glob("csv/Demographic_Variables*/*.csv"))

        if demo_files:
            demo_dfs = []
            for f in demo_files:
                if 'readable' not in str(f):
                    df = pd.read_csv(f, low_memory=False)
                    demo_dfs.append(df)
                    logger.info(f"Loaded {f.name}: {len(df)} rows")

            self.demographics_data = pd.concat(demo_dfs, ignore_index=True)
            logger.info(f"Total demographics records: {len(self.demographics_data)}")

            # Merge on SEQN
            self.merged_data = pd.merge(
                self.audiometry_data,
                self.demographics_data,
                on='SEQN',
                how='inner',
                suffixes=('', '_demo')
            )
            logger.info(f"Merged records: {len(self.merged_data)}")
        else:
            logger.warning("No demographics files found, using audiometry data only")
            self.merged_data = self.audiometry_data

        return self.merged_data

    def extract_thresholds(
        self,
        df: Optional[pd.DataFrame] = None,
        frequencies: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """
        Extract pure-tone thresholds from NHANES data.

        Parameters
        ----------
        df : pd.DataFrame, optional
            Data to extract from. Defaults to merged_data
        frequencies : list of int, optional
            Frequencies to extract. Defaults to NHANES_FREQUENCIES

        Returns
        -------
        pd.DataFrame
            DataFrame with threshold columns for each frequency and ear
        """
        if df is None:
            df = self.merged_data

        if df is None:
            raise ValueError("No data loaded. Call load_data() first.")

        if frequencies is None:
            frequencies = NHANES_FREQUENCIES

        # Extract threshold columns
        threshold_cols = {}
        for freq in frequencies:
            if freq in PTA_COLUMNS:
                for ear, col in PTA_COLUMNS[freq].items():
                    if col in df.columns:
                        threshold_cols[f"{freq}_{ear}"] = df[col]

        threshold_df = pd.DataFrame(threshold_cols)

        # Add demographics if available
        if 'RIDAGEYR' in df.columns:
            threshold_df['age'] = df['RIDAGEYR']
        if 'RIAGENDR' in df.columns:
            threshold_df['sex'] = df['RIAGENDR'].map({1: 'male', 2: 'female'})

        # Filter valid thresholds
        threshold_df = threshold_df.replace([np.inf, -np.inf], np.nan)

        # Remove clearly invalid values
        for col in threshold_df.columns:
            if col not in ['age', 'sex']:
                mask = (threshold_df[col] >= THRESHOLD_BOUNDS['min']) & \
                       (threshold_df[col] <= THRESHOLD_BOUNDS['max'])
                threshold_df.loc[~mask, col] = np.nan

        logger.info(f"Extracted thresholds for {len(threshold_df)} participants")
        return threshold_df

    def build_marginal_priors(
        self,
        threshold_df: Optional[pd.DataFrame] = None,
        frequencies: Optional[List[int]] = None,
        bandwidth: Optional[float] = None
    ) -> Dict[int, Dict[str, stats.gaussian_kde]]:
        """
        Build marginal (unconditional) KDE priors for each frequency.

        Parameters
        ----------
        threshold_df : pd.DataFrame, optional
            DataFrame with threshold columns
        frequencies : list of int, optional
            Frequencies to build priors for
        bandwidth : float, optional
            KDE bandwidth. If None, uses Scott's rule

        Returns
        -------
        dict
            Nested dict: {frequency: {ear: kde_object}}
        """
        if threshold_df is None:
            threshold_df = self.extract_thresholds()

        if frequencies is None:
            frequencies = NHANES_FREQUENCIES

        marginal_priors = {}

        for freq in frequencies:
            marginal_priors[freq] = {}

            for ear in ['right', 'left']:
                col = f"{freq}_{ear}"
                if col not in threshold_df.columns:
                    continue

                values = threshold_df[col].dropna().values

                if len(values) < 10:
                    logger.warning(f"Insufficient data for {freq} Hz {ear} ear")
                    continue

                try:
                    if bandwidth:
                        kde = stats.gaussian_kde(values, bw_method=bandwidth)
                    else:
                        kde = stats.gaussian_kde(values)

                    marginal_priors[freq][ear] = kde
                    logger.info(f"Built KDE for {freq} Hz {ear}: n={len(values)}, bw={kde.factor:.3f}")

                except Exception as e:
                    logger.error(f"Failed to build KDE for {freq} Hz {ear}: {e}")

        self.priors['marginal'] = marginal_priors
        return marginal_priors

    def build_stratified_priors(
        self,
        threshold_df: Optional[pd.DataFrame] = None,
        frequencies: Optional[List[int]] = None,
        age_groups: Optional[List[Tuple[int, int]]] = None
    ) -> Dict:
        """
        Build age-sex stratified priors (Level 1 conditioning).

        Parameters
        ----------
        threshold_df : pd.DataFrame, optional
            DataFrame with threshold columns
        frequencies : list of int, optional
            Frequencies to build priors for
        age_groups : list of tuple, optional
            Age group boundaries as (min, max) tuples

        Returns
        -------
        dict
            Nested dict: {age_group: {sex: {frequency: {ear: kde}}}}
        """
        if threshold_df is None:
            threshold_df = self.extract_thresholds()

        if frequencies is None:
            frequencies = NHANES_FREQUENCIES

        if age_groups is None:
            age_groups = AGE_GROUPS

        stratified_priors = {}

        for age_min, age_max in age_groups:
            age_key = f"{age_min}-{age_max}"
            stratified_priors[age_key] = {}

            for sex in ['male', 'female']:
                stratified_priors[age_key][sex] = {}

                # Filter data
                mask = (threshold_df['age'] >= age_min) & \
                       (threshold_df['age'] <= age_max) & \
                       (threshold_df['sex'] == sex)
                stratum_df = threshold_df[mask]

                if len(stratum_df) < 20:
                    logger.warning(f"Small sample for {age_key} {sex}: n={len(stratum_df)}")
                    continue

                for freq in frequencies:
                    stratified_priors[age_key][sex][freq] = {}

                    for ear in ['right', 'left']:
                        col = f"{freq}_{ear}"
                        if col not in stratum_df.columns:
                            continue

                        values = stratum_df[col].dropna().values

                        if len(values) < 10:
                            continue

                        try:
                            kde = stats.gaussian_kde(values)
                            stratified_priors[age_key][sex][freq][ear] = kde

                        except Exception as e:
                            logger.warning(f"KDE failed for {age_key}/{sex}/{freq} Hz: {e}")

                logger.info(f"Built priors for {age_key} {sex}: n={len(stratum_df)}")

        self.priors['stratified'] = stratified_priors
        return stratified_priors

    def compute_summary_statistics(
        self,
        threshold_df: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Compute summary statistics for threshold distributions.

        Useful for validating data and understanding population distributions.

        Returns
        -------
        pd.DataFrame
            Summary statistics by frequency, ear, age group, and sex
        """
        if threshold_df is None:
            threshold_df = self.extract_thresholds()

        results = []

        for age_min, age_max in AGE_GROUPS:
            for sex in ['male', 'female']:
                mask = (threshold_df['age'] >= age_min) & \
                       (threshold_df['age'] <= age_max) & \
                       (threshold_df['sex'] == sex)
                stratum = threshold_df[mask]

                for freq in NHANES_FREQUENCIES:
                    for ear in ['right', 'left']:
                        col = f"{freq}_{ear}"
                        if col not in stratum.columns:
                            continue

                        values = stratum[col].dropna()

                        results.append({
                            'age_group': f"{age_min}-{age_max}",
                            'sex': sex,
                            'frequency': freq,
                            'ear': ear,
                            'n': len(values),
                            'mean': values.mean() if len(values) > 0 else np.nan,
                            'std': values.std() if len(values) > 0 else np.nan,
                            'median': values.median() if len(values) > 0 else np.nan,
                            'q25': values.quantile(0.25) if len(values) > 0 else np.nan,
                            'q75': values.quantile(0.75) if len(values) > 0 else np.nan,
                        })

        return pd.DataFrame(results)

    def save_priors(self, filename: str = "nhanes_priors.pkl") -> Path:
        """
        Save computed priors to file.

        Parameters
        ----------
        filename : str
            Output filename

        Returns
        -------
        Path
            Path to saved file
        """
        output_path = self.output_dir / filename

        with open(output_path, 'wb') as f:
            pickle.dump(self.priors, f)

        logger.info(f"Saved priors to {output_path}")
        return output_path

    def load_priors(self, filename: str = "nhanes_priors.pkl") -> Dict:
        """
        Load previously computed priors from file.

        Parameters
        ----------
        filename : str
            Input filename

        Returns
        -------
        dict
            Loaded priors dictionary
        """
        input_path = self.output_dir / filename

        with open(input_path, 'rb') as f:
            self.priors = pickle.load(f)

        logger.info(f"Loaded priors from {input_path}")
        return self.priors

    def build_all_priors(self) -> Dict:
        """
        Build all prior distributions (marginal and stratified).

        Convenience method to run full prior construction pipeline.

        Returns
        -------
        dict
            All computed priors
        """
        # Load data if not already loaded
        if self.merged_data is None:
            self.load_data()

        # Extract thresholds
        threshold_df = self.extract_thresholds()

        # Build marginal priors
        self.build_marginal_priors(threshold_df)

        # Build stratified priors
        self.build_stratified_priors(threshold_df)

        # Compute and save summary statistics
        stats_df = self.compute_summary_statistics(threshold_df)
        stats_path = self.output_dir / "threshold_statistics.csv"
        stats_df.to_csv(stats_path, index=False)
        logger.info(f"Saved statistics to {stats_path}")

        # Save priors
        self.save_priors()

        return self.priors


def get_threshold_prior(
    age: int,
    sex: str,
    frequency: int,
    ear: str = 'right',
    priors_path: Optional[Path] = None,
    grid: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Get threshold prior probability distribution for an individual.

    Parameters
    ----------
    age : int
        Age in years
    sex : str
        'male' or 'female'
    frequency : int
        Test frequency in Hz
    ear : str
        'right' or 'left'
    priors_path : Path, optional
        Path to saved priors file
    grid : np.ndarray, optional
        dB HL grid to evaluate prior on. Defaults to -10 to 120 in 1 dB steps

    Returns
    -------
    np.ndarray
        Prior probability density evaluated on grid
    """
    if grid is None:
        grid = np.arange(-10, 121, 1).astype(float)

    # Load priors
    if priors_path is None:
        project_root = Path(__file__).parent.parent.parent
        priors_path = project_root / "data" / "nhanes" / "priors" / "nhanes_priors.pkl"

    if not priors_path.exists():
        logger.warning(f"Priors file not found: {priors_path}")
        # Return uniform prior
        prior = np.ones_like(grid, dtype=float)
        return prior / prior.sum()

    with open(priors_path, 'rb') as f:
        priors = pickle.load(f)

    # Determine age group
    age_group = None
    for age_min, age_max in AGE_GROUPS:
        if age_min <= age <= age_max:
            age_group = f"{age_min}-{age_max}"
            break

    if age_group is None:
        age_group = "18-39" if age < 18 else "80-120"

    # Get appropriate prior
    try:
        if 'stratified' in priors and age_group in priors['stratified']:
            if sex in priors['stratified'][age_group]:
                if frequency in priors['stratified'][age_group][sex]:
                    if ear in priors['stratified'][age_group][sex][frequency]:
                        kde = priors['stratified'][age_group][sex][frequency][ear]
                        prior = kde.evaluate(grid)
                        return prior / prior.sum()

        # Fall back to marginal prior
        if 'marginal' in priors and frequency in priors['marginal']:
            if ear in priors['marginal'][frequency]:
                kde = priors['marginal'][frequency][ear]
                prior = kde.evaluate(grid)
                return prior / prior.sum()

    except Exception as e:
        logger.warning(f"Error getting prior: {e}")

    # Return uniform prior if all else fails
    prior = np.ones_like(grid, dtype=float)
    return prior / prior.sum()
