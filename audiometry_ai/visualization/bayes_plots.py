import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def plot_audiogram(thresholds, uncertainties, true_thresholds, title="Bayesian Audiogram"):
    """
    Plot audiogram with both estimated and true thresholds.
    
    Args:
        thresholds: Dictionary of estimated thresholds
        uncertainties: Dictionary of threshold uncertainties
        true_thresholds: Dictionary of true thresholds
        title: Plot title
    """
    frequencies = sorted(thresholds.keys())
    est_levels = [thresholds[f] for f in frequencies]
    true_levels = [true_thresholds[f] for f in frequencies]
    errors = [uncertainties[f] for f in frequencies]
    
    plt.figure(figsize=(10, 6))
    
    # Plot true thresholds
    plt.plot(frequencies, true_levels, 'o-', color='gray', label='True Threshold', alpha=0.7)
    
    # Plot estimated thresholds with uncertainty
    plt.errorbar(frequencies, est_levels, yerr=errors, fmt='bo-', capsize=5, 
                label='Estimated Threshold', alpha=0.7)
    
    plt.xscale('log')
    plt.xticks(frequencies, frequencies)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Hearing Level (dB)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.gca().invert_yaxis()
    plt.show()

def plot_pdf_evolution(pdf_history, db_range, frequency, progression):
    """Plot evolution of probability distribution for a specific frequency."""
    plt.figure(figsize=(12, 8))
    
    # Create custom colormap from white to blue
    colors = [(1, 1, 1), (0, 0, 1)]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_bins)
    
    # Plot PDF evolution as heatmap
    plt.imshow(
        np.array(pdf_history).T,
        aspect='auto',
        extent=[0, len(pdf_history), db_range[0], db_range[-1]],
        origin='lower',
        cmap=cmap
    )
    
    # Plot test levels
    test_levels = [p[0] for p in progression]
    responses = [p[1] for p in progression]
    for i, (level, response) in enumerate(zip(test_levels, responses)):
        color = 'g' if response else 'r'
        plt.plot(i, level, color + 'o')
    
    plt.colorbar(label='Probability Density')
    plt.title(f'PDF Evolution for {frequency} Hz')
    plt.xlabel('Trial Number')
    plt.ylabel('Hearing Level (dB)')
    plt.show()

def plot_pdf_evolution_gaussians(pdf_history, db_range, frequency, progression, n_trials_to_show=None):
    """
    Plot evolution of probability distribution as series of vertical Gaussians.
    
    Args:
        pdf_history: List of PDFs over time
        db_range: Range of dB values
        frequency: Test frequency
        progression: List of test progression data
        n_trials_to_show: Number of trials to show (None for all)
    """
    plt.figure(figsize=(12, 8))
    
    # Select trials to show
    if n_trials_to_show is None:
        n_trials_to_show = len(pdf_history)
    step = max(1, len(pdf_history) // n_trials_to_show)
    selected_pdfs = pdf_history[::step]
    selected_progression = progression[::step]
    
    # Create colormap from light to dark blue
    colors = plt.cm.Blues(np.linspace(0.2, 1, len(selected_pdfs)))
    
    # Calculate maximum density for scaling
    max_density = max(max(pdf) for pdf in selected_pdfs)
    
    # Determine plot width needed for PDFs
    pdf_width = 0.8  # Width of PDF relative to trial spacing
    
    # Plot each PDF vertically
    for i, (pdf, color, prog) in enumerate(zip(selected_pdfs, colors, selected_progression)):
        # Scale PDF for visualization
        scaled_pdf = pdf / max_density * pdf_width
        
        # Plot vertical PDF
        plt.plot(i + scaled_pdf, db_range, color=color, alpha=0.7)
        
        # Add test point
        level, response = prog[0], prog[1]
        marker = 'go' if response else 'ro'
        plt.plot(i, level, marker, markersize=8, alpha=0.7)
    
    # Customize plot
    plt.grid(True, alpha=0.3)
    plt.title(f'PDF Evolution for {frequency} Hz (Gaussian View)')
    plt.ylabel('Hearing Level (dB)')
    plt.xlabel('Trial Number')
    
    # Add legend for markers
    plt.plot([], [], 'go', label='Response')
    plt.plot([], [], 'ro', label='No Response')
    plt.legend()
    
    # Set x-axis limits to include space for final PDF
    plt.xlim(-0.5, len(selected_pdfs) - 0.5 + pdf_width)
    plt.xticks(range(len(selected_pdfs)), 
               [f'Trial {i*step + 1}' for i in range(len(selected_pdfs))])
    
    # Set y-axis limits to match db_range
    plt.ylim(db_range[0], db_range[-1])
    
    # Invert y-axis to match audiogram convention
    plt.gca().invert_yaxis()
    
    plt.show()

def plot_final_pdfs(posteriors, db_range, thresholds, frequencies=None):
    """
    Plot final PDFs for all test frequencies.
    
    Args:
        posteriors (dict): Dictionary mapping frequencies to final posterior distributions
        db_range (array): Range of dB values used in the test
        thresholds (dict): Dictionary mapping frequencies to estimated thresholds
        frequencies (list, optional): List of frequencies to plot. If None, uses all frequencies in posteriors
    """
    plt.figure(figsize=(12, 8))
    
    if frequencies is None:
        frequencies = sorted(posteriors.keys())
    
    # Create custom colormap from light to dark blue
    colors = plt.cm.Blues(np.linspace(0.4, 1, len(frequencies)))
    
    # Calculate maximum density for scaling
    max_density = max(max(pdf) for pdf in posteriors.values())
    pdf_width = 0.4  # Width of PDF relative to frequency spacing
    
    # Convert frequencies to log scale positions for plotting
    log_freqs = np.log2(frequencies)
    plot_positions = log_freqs - min(log_freqs)
    spacing = 1.0  # Space between frequencies in plot units
    plot_positions = plot_positions * spacing
    
    # Plot each frequency's PDF
    for freq, pos, color in zip(frequencies, plot_positions, colors):
        # Get and scale the PDF
        pdf = posteriors[freq]
        scaled_pdf = pdf / max_density * pdf_width
        
        # Plot vertical PDF
        plt.plot(pos + scaled_pdf, db_range, color=color, alpha=0.7, 
                label=f'{freq} Hz')
        
        # Plot threshold point
        threshold = thresholds[freq]
        plt.plot(pos, threshold, 'ko', markersize=8)
    
    # Customize plot
    plt.grid(True, alpha=0.3)
    plt.title('Final Posterior Distributions Across Frequencies')
    plt.ylabel('Hearing Level (dB)')
    plt.xlabel('Frequency (Hz)')
    
    # Set x-axis ticks to show frequencies
    plt.xticks(plot_positions, frequencies)
    
    # Set y-axis limits and invert
    plt.ylim(db_range[0], db_range[-1])
    plt.gca().invert_yaxis()
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    return plt.gca()