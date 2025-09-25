from hearing_level_gen import *

def respond(test_level, true_level):
    """Simulates a patient's response to a tone presentation.

    Args:
        test_level (float): Tone level in dB HL.
        true_level (float): Patient's true hearing level in dB HL.

    Returns:
        bool: True if the patient responds, False otherwise.
    """
    return test_level >= true_level

def modified_hughson_westlake_procedure(test_frequencies, hearing_profile_data,
                                        pure_tone_sample_params, random_state=None):
    """
    Performs the modified Hughson-Westlake procedure for pure-tone audiometry.
    This procedure adapts the standard audiometric testing by considering initial
    non-responses and detailed progression tracking.

    Args:
        test_frequencies (list): List of frequencies (Hz) to be tested.
        hearing_profile_data (dict): Dictionary containing hearing profile data.
        pure_tone_sample_params (dict): Parameters for generating pure tone samples.
        random_state (int, optional): Seed for the random number generator to ensure reproducibility.

    Returns:
        tuple: A tuple containing the familiarization level, thresholds at each test frequency,
               and progression patterns detailing the testing steps and outcomes.
    """
    thresholds = {}
    progression_patterns = {}
    familiarization_level = None  # Initialize familiarization level for the first frequency tested.

    def find_threshold(frequency, hearing_profile_data, pure_tone_sample_params,
                       familiarization_level, random_state=random_state):
        """
        Finds the hearing threshold for a specific frequency using the modified Hughson-Westlake method.
        Includes error handling for missing frequency data.

        Args:
            frequency (int): The test frequency in Hz.
            hearing_profile_data (dict): Data containing hearing thresholds.
            pure_tone_sample_params (dict): Parameters for generating samples.
            familiarization_level (float): The starting level of testing based on previous tests.
            random_state (int): Seed for random operations to ensure reproducibility.

        Returns:
            tuple: Returns the current familiarization level, determined threshold, and the progression pattern.
        """
        try:
            samples_i = pure_tone_sample(hearing_profile_data, frequency,
                                         **pure_tone_sample_params,
                                         random_state=random_state)
        except KeyError:
            print(f"Warning: Frequency {frequency} not found in hearing_profile_data. Skipping.")
            return familiarization_level, None, []

        if familiarization_level is None:
            test_level = 40  # Starting level for initial familiarization.
            true_level = samples_i.mean()
            while not respond(test_level, true_level):
                test_level += 10
                if test_level > 80:
                    test_level += 5  # Increase in smaller increments above 80 dB HL.
            familiarization_level = test_level
        
        test_level = familiarization_level - 10  # Begin testing 10 dB below the familiarization level.

        true_level = samples_i.mean()  # Reset true level for threshold search.
        threshold = None
        responses = {i: 0 for i in range(-20, 121, 5)}
        n_tests = {i: 0 for i in range(-20, 121, 5)}
        progression = []
        threshold_found = False
        initial_no_response = True
        n_attempts = 0
        phase = 'initial_response'

        while not respond(test_level, true_level):
            progression.append((test_level, False, f"{responses[test_level]}/{n_tests[test_level]}", phase))
            if initial_no_response:
                if test_level <= 80:
                    test_level += 10
                else:
                    test_level += 5
            else:
                test_level += 5

        initial_no_response = False
        progression.append((test_level, True, f"{responses[test_level]}/{n_tests[test_level]}", phase))

        while not threshold_found:
            test_level -= 10
            n_attempts += 1
            if n_attempts < len(samples_i):
                true_level = samples_i[n_attempts]

            response = respond(test_level, true_level)
            phase = 'descend'
            progression.append((test_level, response, f"{responses[test_level]}/{n_tests[test_level]}", phase))

            if not response:
                phase = 'ascend'
                while not respond(test_level, true_level):
                    test_level += 5
                    n_attempts += 1
                    if n_attempts < len(samples_i):
                        true_level = samples_i[n_attempts]

                    response = respond(test_level, true_level)
                    n_tests[test_level] += 1
                    if response:
                        responses[test_level] += 1
                    progression.append((test_level, response, f"{responses[test_level]}/{n_tests[test_level]}", phase))

                    if responses[test_level] > 1 and responses[test_level] / n_tests[test_level] > 0.5:
                        threshold = test_level
                        threshold_found = True
                        break

        return familiarization_level, threshold, progression

    for freq in test_frequencies:
        familiarization_level, thresholds[freq], progression_patterns[freq] = \
            find_threshold(freq, hearing_profile_data, pure_tone_sample_params, familiarization_level)

        # Check for significant gaps between octave thresholds and test intermediate frequencies if necessary.
        octave_higher = freq * 2
        octave_lower = freq // 2
        
        if octave_higher in thresholds:
            if abs(thresholds[freq] - thresholds[octave_higher]) >= 20:
                inter_octave_freq = int((freq + octave_higher) / 2)
                if inter_octave_freq not in thresholds:
                    familiarization_level, thresholds[inter_octave_freq], progression_patterns[inter_octave_freq] = \
                        find_threshold(inter_octave_freq, hearing_profile_data,
                                       pure_tone_sample_params, familiarization_level)
        
        if octave_lower in thresholds:
            if abs(thresholds[freq] - thresholds[octave_lower]) >= 20:
                inter_octave_freq = int((freq + octave_lower) / 2)
                if inter_octave_freq not in thresholds:
                    familiarization_level, thresholds[inter_octave_freq], progression_patterns[inter_octave_freq] = \
                        find_threshold(inter_octave_freq, hearing_profile_data,
                                       pure_tone_sample_params, familiarization_level)

    # Retest at 1000 Hz for consistency.
    if 1000 in test_frequencies:
        _, retest_threshold, _ = find_threshold(1000, hearing_profile_data, pure_tone_sample_params, familiarization_level)
        if abs(retest_threshold - thresholds[1000]) > 5:
            print("Retest at 1000 Hz differs by more than 5 dB.")
        thresholds[1000] = min(thresholds[1000], retest_threshold)

    return familiarization_level, thresholds, progression_patterns