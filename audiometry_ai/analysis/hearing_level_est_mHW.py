import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt
"""
Response model for audiometry testing.
Implements psychometric function for hearing responses.
"""

class HearingResponseModel:
    """Models probability of response to pure tone stimulus."""
    
    def __init__(self, slope=0.2, guess_rate=0.01, lapse_rate=0.01, threshold_probability=0.5):
        """
        Initialize the hearing response model.
        
        Args:
            slope (float): Steepness of the psychometric function
            guess_rate (float): Probability of guessing correctly when stimulus is inaudible
            lapse_rate (float): Probability of missing when stimulus is clearly audible
            threshold_probability (float): Desired probability of response at threshold (0-1).
                                        Default 0.5 gives standard psychometric function.
        """
        if not 0 <= threshold_probability <= 1:
            raise ValueError("threshold_probability must be between 0 and 1")
        
        self.slope = slope
        self.guess_rate = guess_rate
        self.lapse_rate = lapse_rate
        self.threshold_probability = threshold_probability
        
        # Calculate bias from threshold probability
        if threshold_probability == 0:
            self.threshold_bias = float('-inf')
        elif threshold_probability == 1:
            self.threshold_bias = float('inf')
        else:
            from scipy.special import logit
            self.threshold_bias = logit(threshold_probability) / self.slope
    
    def get_response_probability(self, stimulus_level, true_threshold):
        """
        Calculate probability of response for given stimulus level.
        
        Args:
            stimulus_level (float): Current stimulus level in dB
            true_threshold (float): True hearing threshold in dB
            
        Returns:
            float: Probability of response between 0 and 1
        """
        x = self.slope * (stimulus_level - true_threshold + self.threshold_bias)
        p = expit(x)  # expit is the logistic function: 1 / (1 + exp(-x))
        return self.guess_rate + (1 - self.guess_rate - self.lapse_rate) * p
    
    def sample_response(self, stimulus_level, true_threshold, random_state=None):
        """
        Generate binary response based on probability model.
        
        Args:
            stimulus_level (float): Current stimulus level in dB
            true_threshold (float): True hearing threshold in dB
            random_state (int, optional): Random seed for reproducibility
            
        Returns:
            bool: True if response, False if no response
        """
        rng = np.random.default_rng(random_state)
        p = self.get_response_probability(stimulus_level, true_threshold)
        return rng.random() < p
    
"""Constants used in audiometry testing."""

# Maximum output levels for different conduction types
AIR_CONDUCTION_MAX_LEVELS = {
    250: 105, 500: 110, 750: 110,
    1000: 120, 1500: 120, 2000: 120, 3000: 120,
    4000: 120, 6000: 110, 8000: 105
}

BONE_CONDUCTION_MAX_LEVELS = {
    250: 25, 500: 55, 750: 55,
    1000: 70, 1500: 70, 2000: 70, 
    3000: 70, 4000: 70
}

# Default testing parameters
DEFAULT_TEST_FREQUENCIES = [1000, 2000, 4000, 8000, 500, 250]
DEFAULT_STARTING_LEVEL = 40
MIN_TEST_LEVEL = -20
MAX_ITERATIONS = 50

# Response model default parameters
DEFAULT_SLOPE = 0.2
DEFAULT_GUESS_RATE = 0.01
DEFAULT_LAPSE_RATE = 0.01

"""
Main audiometry testing implementation using modified Hughson-Westlake procedure.
"""

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
        Adjust level to stay within valid bounds (-10 to max_level).
        
        Args:
            level (int): Proposed test level
            max_level (int): Maximum allowed level for current frequency
            
        Returns:
            int: Adjusted test level
        """
        return max(-10, min(level, max_level))

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
            if sum(sum(n.values()) for n in n_tests.values()) > 50:
                print(f"Warning: Maximum iterations reached at {frequency} Hz")
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
                print("Retest at 1000 Hz differs by more than 5 dB.")
                self.thresholds[1000] = min(self.thresholds[1000], retest_threshold)

        return self.thresholds, self.progression_patterns

"""Example usage of the audiometry testing system."""

def plot_audiogram(thresholds, title="Audiogram"):
    """Plot audiogram from test results."""
    frequencies = sorted(thresholds.keys())
    levels = [thresholds[f] for f in frequencies]
    
    plt.figure(figsize=(10, 6))
    plt.plot(frequencies, levels, 'b-o')
    plt.xscale('log')
    plt.xticks(frequencies, frequencies)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Hearing Level (dB)')
    plt.title(title)
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.show()

def print_progression(progression, frequency):
    """Print detailed progression for a specific frequency."""
    print(f"\nProgression for {frequency} Hz:")
    print("Level | Response | Ratio | Phase")
    print("-" * 40)
    for level, response, ratio, phase in progression:
        print(f"{level:3d} dB | {str(response):5} | {ratio:^7} | {phase}")

def plot_psychometric_comparison(threshold_probabilities=[0.5, 0.7], slope=1, 
                               guess_rate=0, lapse_rate=0):
    """
    Plot psychometric functions with different threshold probabilities.
    
    Args:
        threshold_probabilities (list): List of threshold probabilities to compare
        slope (float): Slope parameter for all functions
        guess_rate (float): Guess rate for all functions
        lapse_rate (float): Lapse rate for all functions
    """
    plt.figure(figsize=(10, 6))
    stimulus_levels = np.linspace(-20, 20, 1000)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(threshold_probabilities)))
    
    for prob, color in zip(threshold_probabilities, colors):
        model = HearingResponseModel(
            slope=slope,
            guess_rate=guess_rate,
            lapse_rate=lapse_rate,
            threshold_probability=prob
        )
        
        probs = [model.get_response_probability(level, 0) for level in stimulus_levels]
        plt.plot(stimulus_levels, probs, color=color, 
                label=f'p={prob:.1f} at threshold')
        
        # Add horizontal line at threshold probability
        plt.axhline(y=prob, color=color, linestyle=':', alpha=0.5)
    
    # Add threshold reference line
    plt.axvline(x=0, color='k', linestyle='--', label='Threshold')
    
    plt.xlabel('Stimulus Level Relative to Threshold (dB)')
    plt.ylabel('Response Probability')
    plt.title('Comparison of Psychometric Functions')
    plt.grid(True)
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.show()

def main():
    # Create simulated hearing profile with mild to moderate hearing loss
    hearing_profile = {
        250: 10,   
        500: 15,   
        1000: 20,  
        2000: 30,  
        4000: 45,  
        8000: 50   
    }

    # Test different threshold probabilities
    threshold_probs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.99]
    all_results = {}
    
    for prob in threshold_probs:
        # Configure response model
        response_params = {
            'slope': 10,      
            'guess_rate': 0.0,  
            'lapse_rate': 0.0,
            'threshold_probability': prob
        }

        # Initialize and run test
        audiometry = ModifiedHughsonWestlakeAudiometry(
            hearing_profile_data=hearing_profile,
            response_model_params=response_params,
            conduction='air',
            test_frequencies=[1000, 2000, 4000, 8000, 500, 250],
            random_state=42
        )

        # Get results
        thresholds, progression_patterns = audiometry.perform_test()
        all_results[prob] = {
            'thresholds': thresholds,
            'progression': progression_patterns,
        }

    # Create figure with three subplots
    plt.figure(figsize=(20, 6))
    
    # Plot audiogram comparisons
    plt.subplot(1, 3, 1)
    frequencies = sorted(hearing_profile.keys())
    true_levels = [hearing_profile[f] for f in frequencies]
    
    # Plot true thresholds
    plt.plot(frequencies, true_levels, 'k--o', label='True', linewidth=2)
    
    # Plot estimated thresholds for each probability
    colors = plt.cm.tab10(np.linspace(0, 1, len(threshold_probs)))
    for prob, color in zip(threshold_probs, colors):
        levels = [all_results[prob]['thresholds'][f] for f in frequencies]
        plt.plot(frequencies, levels, 'o-', color=color, 
                label=f'p={prob:.1f}', alpha=0.7)
    
    plt.xscale('log')
    plt.xticks(frequencies, frequencies)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Hearing Level (dB)')
    plt.title('Audiogram: True vs Estimated\nfor Different Threshold Probabilities')
    plt.grid(True)
    plt.gca().invert_yaxis()
    plt.legend()

    # Plot psychometric functions
    plt.subplot(1, 3, 2)
    stimulus_levels = np.linspace(-5, 5, 1000)
    
    for prob, color in zip(threshold_probs, colors):
        response_model = HearingResponseModel(
            slope=10, guess_rate=0, lapse_rate=0, threshold_probability=prob
        )
        probabilities = [response_model.get_response_probability(level, 0) 
                        for level in stimulus_levels]
        plt.plot(stimulus_levels, probabilities, '-', color=color, 
                label=f'p={prob:.1f}', alpha=0.7)
    
    plt.axvline(x=0, color='k', linestyle='--', label='Threshold')
    plt.xlabel('Stimulus Level Relative to Threshold (dB)')
    plt.ylabel('Response Probability')
    plt.title('Psychometric Functions\nfor Different Threshold Probabilities')
    plt.grid(True)
    plt.legend()

    # Plot error statistics
    plt.subplot(1, 3, 3)
    stats = []
    for prob in threshold_probs:
        diffs = [all_results[prob]['thresholds'][f] - hearing_profile[f] 
                for f in frequencies 
                if all_results[prob]['thresholds'][f] != 'Not Reached']
        stats.append({
            'prob': prob,
            'mean': np.mean(diffs),
            'std': np.std(diffs),
            'max_abs': max(abs(d) for d in diffs)
        })
    
    x = np.arange(len(threshold_probs))
    width = 0.25
    
    plt.bar(x - width, [s['mean'] for s in stats], width, 
            label='Mean Error', color='b', alpha=0.6)
    plt.bar(x, [s['std'] for s in stats], width,
            label='Std Dev', color='r', alpha=0.6)
    plt.bar(x + width, [s['max_abs'] for s in stats], width,
            label='Max Abs Error', color='g', alpha=0.6)
    
    plt.xlabel('Threshold Probability')
    plt.ylabel('Error (dB)')
    plt.title('Error Statistics\nfor Different Threshold Probabilities')
    plt.xticks(x, [f'{p:.1f}' for p in threshold_probs])
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    # Print detailed statistics
    print("\nDetailed Statistics by Threshold Probability:")
    print("-" * 60)
    print("Prob | Mean Error | Std Dev | Max Abs Error | Tests per Freq")
    print("-" * 60)
    
    for prob in threshold_probs:
        diffs = [all_results[prob]['thresholds'][f] - hearing_profile[f] 
                for f in frequencies 
                if all_results[prob]['thresholds'][f] != 'Not Reached']
        
        # Calculate average number of tests per frequency
        total_tests = sum(len(prog) for prog in all_results[prob]['progression'].values())
        avg_tests = total_tests / len(frequencies)
        
        print(f"{prob:.1f}  | {np.mean(diffs):+6.1f} dB  | {np.std(diffs):6.1f} | "
              f"{max(abs(d) for d in diffs):6.1f}     | {avg_tests:6.1f}")
    
    # Print example progression for 1000 Hz for each probability
    print("\nExample Progression Patterns for 1000 Hz:")
    for prob in threshold_probs:
        print(f"\nThreshold Probability = {prob:.1f}")
        print_progression(all_results[prob]['progression'][1000], 1000)

if __name__ == "__main__":
    main()