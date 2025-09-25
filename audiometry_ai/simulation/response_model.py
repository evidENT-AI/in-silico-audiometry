"""Response model for audiometry testing."""

import numpy as np
from scipy.special import expit, logit

class HearingResponseModel:
    """Models probability of response to pure tone stimulus."""
    
    def __init__(self, slope=0.2, guess_rate=0.01, lapse_rate=0.01, threshold_probability=0.5):
        """Initialize the hearing response model."""
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
            self.threshold_bias = logit(threshold_probability) / self.slope
    
    def get_response_probability(self, stimulus_level, true_threshold):
        """Calculate probability of response for given stimulus level."""
        x = self.slope * (stimulus_level - true_threshold + self.threshold_bias)
        p = expit(x)
        return self.guess_rate + (1 - self.guess_rate - self.lapse_rate) * p
    
    def sample_response(self, stimulus_level, true_threshold, random_state=None):
        """Generate binary response based on probability model."""
        rng = np.random.default_rng(random_state)
        p = self.get_response_probability(stimulus_level, true_threshold)
        return rng.random() < p