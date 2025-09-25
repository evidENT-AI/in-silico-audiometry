"""Visualization functions for audiometry results."""

import numpy as np
import matplotlib.pyplot as plt

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

def plot_psychometric_comparison(model_class, threshold_probabilities=[0.5, 0.7],
                               slope=1, guess_rate=0, lapse_rate=0):
    """Plot psychometric functions with different threshold probabilities."""
    plt.figure(figsize=(10, 6))
    stimulus_levels = np.linspace(-20, 20, 1000)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(threshold_probabilities)))
    
    for prob, color in zip(threshold_probabilities, colors):
        model = model_class(
            slope=slope,
            guess_rate=guess_rate,
            lapse_rate=lapse_rate,
            threshold_probability=prob
        )
        
        probs = [model.get_response_probability(level, 0) for level in stimulus_levels]
        plt.plot(stimulus_levels, probs, color=color, 
                label=f'p={prob:.1f} at threshold')
        
        plt.axhline(y=prob, color=color, linestyle=':', alpha=0.5)
    
    plt.axvline(x=0, color='k', linestyle='--', label='Threshold')
    
    plt.xlabel('Stimulus Level Relative to Threshold (dB)')
    plt.ylabel('Response Probability')
    plt.title('Comparison of Psychometric Functions')
    plt.grid(True)
    plt.legend()
    plt.ylim(-0.05, 1.05)
    plt.show()