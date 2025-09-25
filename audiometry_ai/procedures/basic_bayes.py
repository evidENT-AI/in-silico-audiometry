"""
Main audiometry testing implementation using basic Bayesian procedure.
"""
from typing import Dict, List, Tuple, Union, Optional
import numpy as np
from scipy.stats import norm
from scipy.signal import convolve
from enum import Enum

# Local imports
from ..simulation.response_model import HearingResponseModel
from ..utils.defaults import (
    AIR_CONDUCTION_MAX_LEVELS,
    BONE_CONDUCTION_MAX_LEVELS,
    DEFAULT_TEST_FREQUENCIES,
    DEFAULT_STARTING_LEVEL,
    DEFAULT_EXPERT_STARTING_LEVEL,
    MIN_TEST_LEVEL,
    MAX_ITERATIONS,
    EXPERT_MAX_ITERATIONS
)

class BayesianPureToneAudiometry:
    class TestPhase(Enum):
        INITIAL = 'initial'
        ASCENDING = 'ascending'
        DESCENDING = 'descending'

    def __init__(
        self,
        hearing_profile_data: dict,
        response_model_params: Optional[dict] = None,
        test_frequencies: Optional[List[int]] = None,
        max_trials_per_freq: int = 20,
        convergence_threshold_db: float = 5.0,
        grid_resolution_db: float = 1.0,
        fp_rate: float = 0.02,  # False positive rate
        fn_rate: float = 0.05,  # False negative rate
        ascending_weight: float = 1.0,  # Relative weight for ascending trials
        descending_weight: float = 1.0,  # Relative weight for descending trials
        random_state: Optional[int] = None
    ):
        """
        Initialize enhanced Bayesian pure-tone audiometry testing.
        """
        self.hearing_profile_data = hearing_profile_data
        self.response_model = HearingResponseModel(**(response_model_params or {}))
        self.test_frequencies = test_frequencies or [250, 500, 1000, 2000, 4000, 8000]
        self.max_trials = max_trials_per_freq
        self.convergence_threshold = convergence_threshold_db
        self.grid_resolution = grid_resolution_db
        self.fp_rate = fp_rate
        self.fn_rate = fn_rate
        self.ascending_weight = ascending_weight
        self.descending_weight = descending_weight
        self.rng = np.random.default_rng(random_state)
        
        # Initialize probability grids
        self.db_range = np.arange(-10, 121, self.grid_resolution)
        self.priors = {}
        self.posteriors = {}
        self.thresholds = {}
        self.uncertainties = {}
        self.progression_patterns = {}
        self.pdf_history = {}  # Track PDF evolution
        
    def _get_octave_influence(self, freq1: int, freq2: int) -> float:
        """
        Calculate influence weight based on octave distance between frequencies.
        Uses continuous decay rather than discrete categories.
        """
        octave_diff = abs(np.log2(freq1 / freq2))
        return np.exp(-octave_diff)  # Exponential decay with octave difference
        
    def _initialize_prior(self, frequency: int) -> np.ndarray:
        """
        Initialize prior with influence from all previously tested frequencies.
        """
        # Start with uniform prior
        prior = np.ones_like(self.db_range) / len(self.db_range)
        
        # Get all previously tested frequencies
        tested_freqs = list(self.posteriors.keys())
        if not tested_freqs:
            return prior
            
        # Combine influences from all previous frequencies
        influence = np.zeros_like(prior)
        total_weight = 0
        
        for tested_freq in tested_freqs:
            weight = self._get_octave_influence(frequency, tested_freq)
            total_weight += weight
            influence += weight * self.posteriors[tested_freq]
            
        if total_weight > 0:
            prior = (influence / total_weight + prior) / 2
            
        return prior / prior.sum()
        
    def _get_initial_level(self, prior: np.ndarray) -> float:
        """
        Determine initial test level based on prior distribution.
        """
        if np.allclose(prior, prior[0]):  # Uniform distribution
            return 40.0
            
        # Get weighted average of prior
        expected_threshold = np.sum(self.db_range * prior)
        
        # Apply clinical constraints
        if expected_threshold > 80:
            return 80.0
        return min(max(40.0, expected_threshold), 80.0)
        
    def _apply_step_constraints(
        self,
        current_level: float,
        proposed_level: float
    ) -> float:
        """
        Apply clinical step size constraints.
        """
        if current_level >= 80:
            # Already above 80, limit to 5 dB steps
            return current_level + np.clip(proposed_level - current_level, -5, 5)
        elif proposed_level > 80:
            # Moving into high-level region
            return min(85, current_level + 20)
        elif current_level < 80 and (proposed_level - current_level) > 20:
            # Below 80 but step too large
            return current_level + 20
        return proposed_level
        
    def _get_response_probability(
        self,
        stimulus_level: float,
        threshold: float,
        slope: float = 0.1
    ) -> float:
        """
        Get probability of response given stimulus level and threshold.
        Uses psychometric function (cumulative normal).
        """
        x = stimulus_level - threshold
        return norm.cdf(x * slope)
        
    def _get_entropy(self, distribution: np.ndarray) -> float:
        """Calculate Shannon entropy of distribution."""
        positive_probs = distribution[distribution > 0]
        return -np.sum(positive_probs * np.log2(positive_probs))
        
    def _estimate_threshold(self, posterior: np.ndarray) -> Tuple[float, float]:
        """
        Estimate threshold and uncertainty from posterior distribution.
        """
        threshold = np.sum(self.db_range * posterior)
        variance = np.sum(posterior * (self.db_range - threshold)**2)
        return threshold, np.sqrt(variance)

    def _update_posterior_with_error_rates(
        self,
        prior: np.ndarray,
        test_level: float,
        response: bool,
        phase: TestPhase
    ) -> np.ndarray:
        """
        Update posterior accounting for false positive/negative rates.
        """
        likelihoods = np.zeros_like(self.db_range)
        
        for i, threshold in enumerate(self.db_range):
            # Base response probability from psychometric function
            p_base = self._get_response_probability(test_level, threshold)
            
            # Adjust for false positives/negatives
            p_response = p_base * (1 - self.fn_rate) + (1 - p_base) * self.fp_rate
            
            # Apply phase-specific weighting
            if phase == self.TestPhase.ASCENDING:
                p_response *= self.ascending_weight
            elif phase == self.TestPhase.DESCENDING:
                p_response *= self.descending_weight
                
            likelihoods[i] = p_response if response else (1 - p_response)
            
        posterior = likelihoods * prior
        return posterior / posterior.sum()
        
    def _get_optimal_test_level_for_phase(
        self,
        posterior: np.ndarray,
        previous_level: Optional[float],
        phase: TestPhase
    ) -> float:
        """
        Get optimal test level considering test phase.
        """
        if phase == self.TestPhase.INITIAL:
            # Aggressive stepping to find first response
            if previous_level is None:
                return self._get_initial_level(posterior)
            elif previous_level < 80:
                return min(80, previous_level + 20)
            else:
                return min(120, previous_level + 5)
                
        # Calculate expected information gain for each possible level
        info_gains = np.zeros_like(self.db_range)
        
        for i, test_level in enumerate(self.db_range):
            # Skip invalid levels due to step constraints
            if previous_level is not None:
                constrained_level = self._apply_step_constraints(previous_level, test_level)
                if constrained_level != test_level:
                    continue
                    
            # Calculate expected information gain
            p_response = 0
            for threshold, prob in zip(self.db_range, posterior):
                response_prob = self._get_response_probability(test_level, threshold)
                # Adjust for errors
                response_prob = response_prob * (1 - self.fn_rate) + (1 - response_prob) * self.fp_rate
                p_response += prob * response_prob
                
            # Calculate expected entropy reduction
            h_current = self._get_entropy(posterior)
            h_yes = self._get_entropy(
                self._update_posterior_with_error_rates(
                    posterior, test_level, True, phase))
            h_no = self._get_entropy(
                self._update_posterior_with_error_rates(
                    posterior, test_level, False, phase))
                    
            info_gains[i] = h_current - (p_response * h_yes + (1 - p_response) * h_no)
            
        return self.db_range[np.argmax(info_gains)]
        
    def _test_frequency(self, frequency: int) -> Tuple[float, float, List]:
        """
        Perform Bayesian threshold estimation for single frequency.
        """
        prior = self._initialize_prior(frequency)
        posterior = prior
        progression = []
        pdf_history = []  # Track PDF evolution
        prev_level = None
        phase = self.TestPhase.INITIAL
        initial_response_found = False
        
        for trial in range(self.max_trials):
            # Get optimal test level for current phase
            proposed_level = self._get_optimal_test_level_for_phase(
                posterior, prev_level, phase)
            
            # Apply clinical constraints
            test_level = self._apply_step_constraints(
                prev_level if prev_level is not None else proposed_level,
                proposed_level)
            
            # Get response
            response = self.response_model.sample_response(
                test_level,
                self.hearing_profile_data[frequency],
                random_state=self.rng
            )
            
            # Update phase based on response
            if phase == self.TestPhase.INITIAL and response:
                phase = self.TestPhase.DESCENDING
                initial_response_found = True
            elif phase == self.TestPhase.DESCENDING and not response:
                phase = self.TestPhase.ASCENDING
            elif phase == self.TestPhase.ASCENDING and response:
                phase = self.TestPhase.DESCENDING
                
            # Update posterior with error rates
            posterior = self._update_posterior_with_error_rates(
                posterior, test_level, response, phase)
                
            # Record progression and PDF
            threshold_est, uncertainty = self._estimate_threshold(posterior)
            progression.append((test_level, response, threshold_est, uncertainty, phase.value))
            pdf_history.append(posterior.copy())
            
            # Check convergence
            if (initial_response_found and 
                uncertainty < self.convergence_threshold):
                break
                
            prev_level = test_level
            
        return threshold_est, uncertainty, progression, pdf_history
        
    def perform_test(self) -> Dict:
        """
        Perform complete audiometry test using enhanced Bayesian approach.
        """
        for freq in self.test_frequencies:
            threshold, uncertainty, progression, pdf_history = self._test_frequency(freq)
            
            self.thresholds[freq] = threshold
            self.uncertainties[freq] = uncertainty
            self.progression_patterns[freq] = progression
            self.pdf_history[freq] = pdf_history
            
            # Store final posterior for future frequencies
            if freq in self.posteriors:
                del self.posteriors[freq]
            self.posteriors[freq] = pdf_history[-1]
            
        return {
            'thresholds': self.thresholds,
            'uncertainties': self.uncertainties,
            'progression_patterns': self.progression_patterns,
            'pdf_history': self.pdf_history
        }