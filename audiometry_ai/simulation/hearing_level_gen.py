import numpy as np
import pandas as pd
from scipy.stats import truncnorm

def generate_clipped_data(n_points=8000, mu=10, variance=0.05, data_min=0,
                          data_max=20, seed=None):
    """
    Generates data distributed according to a truncated normal distribution,
    then clips the data to a specified range.

    Args:
        n_points (int): Number of data points to generate.
        mu (float): Mean of the truncated normal distribution.
        variance (float): Variance of the truncated normal distribution.
        data_min (float): Minimum value to clip the data to.
        data_max (float): Maximum value to clip the data to.
        seed (int): Random seed for reproducibility.

    Returns:
        x_data (np.ndarray): The x data points, linearly spaced.
        data_clipped (np.ndarray): The generated data, clipped to the specified range.
    """
    np.random.seed(seed)
    x_data = np.linspace(0, n_points, n_points)
    sigma = np.sqrt(variance)  # Standard deviation (sqrt of variance)

    # Generate data
    data = truncnorm.rvs((data_min - mu) / sigma, (data_max - mu) / sigma,
                         loc=mu, scale=sigma, size=x_data.shape)

    # Clip data
    data_clipped = np.clip(data, data_min, data_max)

    return x_data, data_clipped

def generate_audio_profile_dataframe(frequencies, n_points=8000, mu_right=10, mu_left=10,
                       variance_right=0.05, variance_left=0.05,
                       data_min=0, data_max=20, seed=None):
    """
    Generates a pandas DataFrame with synthetic data of a audiometric
    profile for the specified frequencies.

    Args:
        frequencies (list): List of frequencies (in Hz) to extract data for.
        n_points (int): Number of data points to generate.
        mu_right (float): Mean of the truncated normal distribution for right ear.
        mu_left (float): Mean of the truncated normal distribution for left ear.
        variance_right (float): Variance of the truncated normal distribution for right ear.
        variance_left (float): Variance of the truncated normal distribution for left ear.
        data_min (float): Minimum value to clip the data to.
        data_max (float): Maximum value to clip the data to.
        seed (int): Random seed for reproducibility.

    Returns:
        dataframe (pd.DataFrame): The generated DataFrame with the specified frequencies.
    """
    # Generate data for right ear
    _, data_clipped_right = generate_clipped_data(
        n_points, mu_right, variance_right, data_min, data_max, seed=seed)

    # Generate data for left ear
    _, data_clipped_left = generate_clipped_data(
        n_points, mu_left, variance_left, data_min, data_max, seed=seed+1)

    # Create dictionary to store data for each frequency
    data_dict = {}

    # Extract data for each frequency and store in dictionary
    for freq in frequencies:
        index = freq - 1
        data_dict[f"{freq/1000:.1f}kHz Right"] = data_clipped_right[index]
        data_dict[f"{freq/1000:.1f}kHz Left"] = data_clipped_left[index]

    # Create DataFrame from the data dictionary
    dataframe = pd.DataFrame(data_dict, index=[0])

    return dataframe

def synth_to_dataframe(frequencies, data_clipped_right, data_clipped_left):
    """
    Converts exsisting synth data to a pandas DataFrame for the specified frequencies.

    Args:
        frequencies (list): List of frequencies (in Hz) to extract data for.
        data_clipped_right (np.ndarray): The clipped data for the right ear.
        data_clipped_left (np.ndarray): The clipped data for the left ear.

    Returns:
        dataframe (pd.DataFrame): The generated DataFrame with the specified frequencies.
    """
    # Create dictionary to store data for each frequency
    data_dict = {}

    # Extract data for each frequency and store in dictionary
    for freq in frequencies:
        index = freq - 1
        data_dict[f"{freq/1000:.1f}kHz Right"] = data_clipped_right[index]
        data_dict[f"{freq/1000:.1f}kHz Left"] = data_clipped_left[index]

    # Create DataFrame from the data dictionary
    dataframe = pd.DataFrame(data_dict, index=[0])

    return dataframe

def modulate_severity(hearing_level_data, gain):
    """Modulates the severity of hearing level data by a specified gain.

    This function applies a constant gain to each element in the hearing
    level data array to simulate an increase or decrease in severity.

    Args:
        hearing_level_data (np.ndarray): The original array of hearing level data.
        gain (float): The gain to be applied to each element in the hearing level data.

    Returns:
        np.ndarray: The modulated hearing level data with the gain applied.
    """
    modulated_data = hearing_level_data + gain
    return modulated_data

def modulate_monotonic(hearing_level_data, positive, gain, n_samples):
    """Modulates hearing level data with a monotonic change.

    Applies a linear modulation to the hearing level data. The modulation can be either
    increasing or decreasing based on the 'positive' flag. The total change in modulation
    is specified by 'gain', applied linearly across 'n_samples' points.

    Args:
        hearing_level_data (np.ndarray): The original array of hearing level data.
        positive (bool): Flag indicating the direction of modulation. True for an
                         increasing change, False for a decreasing change.
        gain (float): The total change to be applied across the modulation.
        n_samples (int): The number of samples over which to apply the modulation.

    Returns:
        np.ndarray: The modulated hearing level data.
    """
    # Generate a linear space for the modulation, direction based on 'positive'
    monotone = np.linspace(0, gain, n_samples)
    if not positive:
        monotone = monotone[::-1]  # Reverse for decreasing modulation

    # Apply the monotonic modulation to the hearing level data
    modulated_data = hearing_level_data + monotone
    return modulated_data

def modulate_sigmoidal(hearing_level_data, x0, k, gain):
    """Modulates hearing level data with a sigmoidal function.

    This function applies a sigmoidal modulation to the hearing level data. The
    sigmoidal curve is determined by a midpoint `x0`, steepness `k`, and a maximum
    gain. This results in a smooth, nonlinear transition in the hearing level data.

    Args:
        hearing_level_data (np.ndarray): The original array of hearing level data.
        x0 (float): The midpoint of the sigmoid where the transition occurs.
        k (float): The steepness of the sigmoid curve.
        gain (float): The maximum change applied to the data.

    Returns:
        np.ndarray: The array of hearing level data after applying sigmoidal modulation.
    """
    n_samples = len(hearing_level_data)
    x = np.linspace(0, n_samples-1, n_samples)
    sigmoid = 1 / (1 + np.exp(-k * (x - x0)))
    modulated_data = hearing_level_data + gain * sigmoid
    return modulated_data

def pure_tone_sample(hearing_profile_data, pure_tone, noise, n_samples,
                     source='theory', distribution='Gaussian', random_state=None):
    """
    Generate pure tone samples based on the hearing profile data.

    Args:
        hearing_profile_data (list or DataFrame): List or DataFrame of hearing
            thresholds at each frequency.
        pure_tone (float or str): Pure tone frequency in Hz or name of column
            in DataFrame.
        noise (float): Standard deviation of the noise distribution.
        n_samples (int): Number of samples to generate.
        source (str): Source of the hearing profile data ('data' or 'theory').
        distribution (str, optional): Type of distribution to use
            ('Gaussian' or 'Custom'). Defaults to 'Gaussian'.
        random_state (int or None, optional): Random seed for reproducibility.
            If None, a random seed will be used. Defaults to None.

    Returns:
        numpy.ndarray: Array of pure tone samples.

    Raises:
        ValueError: If the specified source is not recognized.
        TypeError: If inputs are of incorrect type.
    """
    import numpy as np
    
    rng = np.random.default_rng(random_state)

    # Type checking for pure_tone
    if isinstance(pure_tone, (list, tuple)):
        raise TypeError("pure_tone must be a single frequency value, not a list or tuple")
    
    # Convert pure_tone to the appropriate type based on source
    if source == 'data':
        if not isinstance(pure_tone, (str, int, float)):
            raise TypeError("pure_tone must be a string, integer, or float for 'data' source")
        true_hl_at_pure_tone = hearing_profile_data[pure_tone]
    elif source == 'theory':
        try:
            index = int(pure_tone) - 1
            if not isinstance(hearing_profile_data, (list, np.ndarray)):
                raise TypeError("hearing_profile_data must be a list or numpy array for 'theory' source")
            true_hl_at_pure_tone = hearing_profile_data[index]
        except (ValueError, TypeError) as e:
            raise TypeError(f"Could not convert pure_tone to index: {e}")
    else:
        raise ValueError(f"Unrecognized source: {source}. Expected 'data' or 'theory'.")

    if distribution == 'Gaussian':
        samples = rng.normal(true_hl_at_pure_tone, noise, size=n_samples)
        return samples
    else:
        raise ValueError(f"Unrecognized distribution: {distribution}. Expected 'Gaussian'.")