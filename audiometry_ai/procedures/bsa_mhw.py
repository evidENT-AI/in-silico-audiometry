"""
Main audiometry testing implementation using modified Hughson-Westlake procedure.
"""
# Standard library imports
from typing import Dict, List, Tuple, Union, Optional

# Third-party imports
import numpy as np

# Local imports
from ..simulation.response_model import HearingResponseModel
from ..utils.defaults import (
    AIR_CONDUCTION_MAX_LEVELS,
    BONE_CONDUCTION_MAX_LEVELS,
    DEFAULT_TEST_FREQUENCIES,
    DEFAULT_STARTING_LEVEL,
    MIN_TEST_LEVEL,
    MAX_ITERATIONS
)

# Type aliases for clarity
ThresholdType = Union[int, float, str]  # Can be number or 'Not Reached'
ProgressionType = List[Tuple[int, bool, str, str]]  # (level, response, ratio, phase)
ThresholdsDict = Dict[int, ThresholdType]
ProgressionDict = Dict[int, ProgressionType]

class ModifiedHughsonWestlakeAudiometry:
    def __init__(self, hearing_profile_data,
                 response_model_params=None,
                 conduction='air',
                 test_frequencies=None,
                 random_state=None):
        """
        Initialize audiometry testing.
        
        Args:
            hearing_profile_data (dict or list): Hearing thresholds by frequency
            response_model_params (dict): Parameters for response model
            conduction (str): 'air' or 'bone'
            test_frequencies (list): Frequencies to test
            random_state (int): Random seed for reproducibility
        """
        self.hearing_profile_data = hearing_profile_data
        self.response_model = HearingResponseModel(**(response_model_params or {}))
        self.conduction = conduction
        self.test_frequencies = test_frequencies or DEFAULT_TEST_FREQUENCIES.copy()
        self.random_state = random_state
        
        # Initialize state
        self.thresholds = {}
        self.progression_patterns = {}
        self.max_levels = {}
        self.rng = np.random.default_rng(random_state)
        
        # Validate hearing profile data
        if isinstance(hearing_profile_data, dict):
            missing_freqs = [f for f in self.test_frequencies if f not in hearing_profile_data]
            if missing_freqs:
                raise ValueError(f"Missing frequencies in hearing profile: {missing_freqs}")
        elif isinstance(hearing_profile_data, list):
            if len(hearing_profile_data) < len(self.test_frequencies):
                raise ValueError("Insufficient threshold values in hearing profile")
        else:
            raise ValueError("hearing_profile_data must be either a dict or list")

    def _respond(self, test_level, frequency):
        """Generate response using psychometric function."""
        true_threshold = self._get_true_threshold(frequency)
        return self.response_model.sample_response(
            stimulus_level=test_level,
            true_threshold=true_threshold,
            random_state=self.rng
        )
    
    def _get_max_levels(self):
        """Set maximum output levels based on conduction type."""
        self.max_levels = (AIR_CONDUCTION_MAX_LEVELS if self.conduction == 'air' 
                          else BONE_CONDUCTION_MAX_LEVELS)
        
    def _get_true_threshold(self, frequency):
        """Get the true threshold for a given frequency."""
        try:
            if isinstance(self.hearing_profile_data, dict):
                return self.hearing_profile_data[frequency]
            return self.hearing_profile_data[int(frequency) - 1]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Cannot get threshold for frequency {frequency}: {str(e)}")

    def _adjust_level(self, level, max_level):
        """
        Adjust level to stay within valid bounds (MIN_TEST_LEVEL to max_level).
        
        Args:
            level (int): Proposed test level
            max_level (int): Maximum allowed level for current frequency
            
        Returns:
            int: Adjusted test level
        """
        return max(MIN_TEST_LEVEL, min(level, max_level))

    def _get_starting_level(self, frequency):
        """
        Determine starting level for a frequency based on previous threshold or familiarization.
        
        Args:
            frequency (int): Test frequency in Hz
            
        Returns:
            int or str: Starting level in dB or 'Not Reached'
        """
        max_level = self.max_levels[frequency]
        
        # Get previous frequency's threshold if not first frequency
        current_index = self.test_frequencies.index(frequency)
        if current_index > 0:
            prev_threshold = self.thresholds.get(self.test_frequencies[current_index-1])
            if isinstance(prev_threshold, (int, float)):
                return self._adjust_level(prev_threshold - 10, max_level)

        # No valid previous threshold, perform familiarization test
        test_level = self._adjust_level(DEFAULT_STARTING_LEVEL, max_level)
        max_level_count = 0
        
        while not self._respond(test_level, frequency):
            if test_level == max_level:
                max_level_count += 1
                if max_level_count >= 2:
                    return 'Not Reached'
            
            if test_level < 80:
                test_level = self._adjust_level(test_level + 10, max_level)
            else:
                test_level = self._adjust_level(test_level + 5, max_level)
        
        return self._adjust_level(test_level - 10, max_level)

    def _find_threshold(self, frequency):
        """
        Finds the hearing threshold for a specific frequency using the modified Hughson-Westlake method.
        
        Args:
            frequency (int): The test frequency in Hz.
            
        Returns:
            tuple: Returns (threshold, progression_pattern)
        """
        max_level = self.max_levels[frequency]
        responses = {}
        n_tests = {}
        progression = []
        phase = 'initial'
        max_level_count = 0

        # Get starting level for this frequency
        test_level = self._get_starting_level(frequency)
        if test_level == 'Not Reached':
            return 'Not Reached', []
        
        # Initial phase - increase until response
        while phase == 'initial':
            response = self._respond(test_level, frequency)
            
            # Track max level presentations
            if test_level == max_level and not response:
                max_level_count += 1
                if max_level_count >= 2:
                    return 'Not Reached', progression
            
            # Update tracking
            if test_level not in responses:
                responses[test_level] = {'ascending': 0, 'descending': 0, 'initial': 0}
                n_tests[test_level] = {'ascending': 0, 'descending': 0, 'initial': 0}
            
            responses[test_level][phase] += 1 if response else 0
            n_tests[test_level][phase] += 1
            
            # Record progression
            ascending_responses = responses[test_level]['ascending']
            ascending_tests = n_tests[test_level]['ascending']
            ratio = f"{ascending_responses}/{ascending_tests}" if ascending_tests > 0 else "0/0"
            progression.append((test_level, response, ratio, phase))

            if response:
                phase = 'descending'
                test_level = self._adjust_level(test_level - 10, max_level)
            else:
                if test_level < 80:
                    test_level = self._adjust_level(test_level + 10, max_level)
                else:
                    test_level = self._adjust_level(test_level + 5, max_level)

        # Main threshold finding loop
        while True:
            response = self._respond(test_level, frequency)
            
            # Track max level presentations
            if test_level == max_level and not response:
                max_level_count += 1
                if max_level_count >= 2:
                    return 'Not Reached', progression
            
            # Update tracking
            if test_level not in responses:
                responses[test_level] = {'ascending': 0, 'descending': 0, 'initial': 0}
                n_tests[test_level] = {'ascending': 0, 'descending': 0, 'initial': 0}
            
            responses[test_level][phase] += 1 if response else 0
            n_tests[test_level][phase] += 1
            
            # Record progression
            ascending_responses = responses[test_level]['ascending']
            ascending_tests = n_tests[test_level]['ascending']
            ratio = f"{ascending_responses}/{ascending_tests}" if ascending_tests > 0 else "0/0"
            progression.append((test_level, response, ratio, phase))

            # Check threshold criteria
            if phase == 'ascending':
                for level in responses:
                    ascending_responses = responses[level]['ascending']
                    ascending_tests = n_tests[level]['ascending']
                    if (ascending_responses >= 2 and
                        ascending_tests > 0 and
                        ascending_responses / ascending_tests > 0.5):
                        return level, progression

            # Determine next level and phase
            if phase == 'descending':
                if response:
                    test_level = self._adjust_level(test_level - 10, max_level)
                else:
                    phase = 'ascending'
                    test_level = self._adjust_level(test_level + 5, max_level)
            else:  # ascending phase
                if response:
                    phase = 'descending'
                    test_level = self._adjust_level(test_level - 10, max_level)
                else:
                    test_level = self._adjust_level(test_level + 5, max_level)

            # Prevent infinite loops
            if sum(sum(n.values()) for n in n_tests.values()) > MAX_ITERATIONS:
                #print(f"Warning: Maximum iterations reached at {frequency} Hz")
                return 'Not Reached', progression

    def perform_test(self):
        """
        Perform complete audiometry test.
        
        Returns:
            tuple: (thresholds, progression_patterns) where:
                - thresholds: Dict mapping frequencies to their thresholds
                - progression_patterns: Dict mapping frequencies to their test progressions
        """
        self._get_max_levels()

        for freq in self.test_frequencies:
            threshold, progression = self._find_threshold(freq)
            self.thresholds[freq] = threshold
            self.progression_patterns[freq] = progression

        # Retest at 1000 Hz
        if 1000 in self.test_frequencies:
            retest_threshold, _ = self._find_threshold(1000)
            
            if (retest_threshold != 'Not Reached' and 
                self.thresholds[1000] != 'Not Reached' and 
                abs(retest_threshold - self.thresholds[1000]) > 5):
                #print("Retest at 1000 Hz differs by more than 5 dB.")
                self.thresholds[1000] = min(self.thresholds[1000], retest_threshold)

        return self.thresholds, self.progression_patterns
