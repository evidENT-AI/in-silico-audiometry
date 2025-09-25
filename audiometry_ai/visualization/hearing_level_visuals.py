import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

def thresholds_to_curve(df, split=False, title=None):
    """
    Generates a subplot of audiogram curves for each participant in the dataframe,
    arranging the subplots in a grid that minimizes the difference between the number of rows and columns.
    If split is True, it creates separate subplots for each ear.

    Args:
    df (pandas.DataFrame): A dataframe where each row represents a participant's
                           hearing thresholds at different frequencies for both ears.
                           Expected columns are '0.5kHz Left', '1kHz Left', '2kHz Left',
                           '4kHz Left', '8kHz Left', '0.5kHz Right', '1kHz Right',
                           '2kHz Right', '4kHz Right', '8kHz Right', with some flexibility in naming.
    split (bool): If True, creates separate subplots for each ear. Default is False.
    title (str): Title for the plot. If None, no title is set. For split view, this will be used as a base title.

    Raises:
    ValueError: If the DataFrame does not contain the expected columns.
    """
    # Define the required frequencies and ears
    freqs = ['0.5', '1', '2', '4', '8']
    ears = ['Right', 'Left']

    # Function to check if a column exists with a specific frequency and ear
    def has_column(freq, ear):
        return any(f"{freq}kHz {ear}" in col or f"{freq}.0kHz {ear}" in col for col in df.columns)

    # Check if the dataframe contains all the required columns
    if not all(has_column(freq, ear) for freq in freqs for ear in ears):
        raise ValueError("DataFrame does not contain all required frequency columns for both ears.")

    # Define color palette
    palette = sns.color_palette("deep")
    red = palette[3]
    blue = palette[0]

    # Determine number of participants and grid layout
    num_participants = df.shape[0]
    if split:
        n_cols = math.ceil(math.sqrt(num_participants * 2))
        n_rows = math.ceil((num_participants * 2) / n_cols)
    else:
        n_cols = math.ceil(math.sqrt(num_participants))
        n_rows = math.ceil(num_participants / n_cols)

    # Create a figure with subplots arranged in a grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), sharex=True, sharey=True)

    # Set the main title if provided
    if title:
        if split:
            fig.suptitle(title, fontsize=16, y=1.02)
        else:
            fig.suptitle(title, fontsize=16)

    # Check if axes is a single Axes object or an array of Axes objects
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    # Plot each participant's data
    for idx, (_, row) in enumerate(df.iterrows()):
        if split:
            ax_right = axes.flat[idx * 2]
            ax_left = axes.flat[idx * 2 + 1]
        else:
            ax = axes.flat[idx]

        for ear, color, marker in [('Right', red, 'X'), ('Left', blue, 'o')]:
            data = []
            sorted_freqs = []
            for freq in freqs:
                for col in df.columns:
                    if f"{freq}kHz {ear}" in col or f"{freq}.0kHz {ear}" in col:
                        data.append(row[col])
                        sorted_freqs.append(freq)
                        break

            if split:
                ax = ax_right if ear == 'Right' else ax_left
                ax.set_title(f'{title} - {ear} Ear' if title else '')
            else:
                ax.set_title(f'{title}' if title else '')

            if marker == 'o':
                ax.plot(sorted_freqs, data, label=f'{ear} Ear', marker=marker, linestyle='-',
                        color=color, markersize=10, lw=2, markeredgewidth=2,
                        markerfacecolor='none', markeredgecolor=color)  # Hollow circle for Left ear
            elif marker == 'X':
                ax.plot(sorted_freqs, data, label=f'{ear} Ear', marker=marker, linestyle='-',
                        color=color, markersize=12, lw=2, markerfacecolor=color,
                        markeredgecolor='white', markeredgewidth=1)
            
            ax.set_xlabel('Frequency (kHz)')
            ax.set_ylabel('Hearing Level (dB)')
            ax.invert_yaxis()
            ax.set_ylim(120, -20)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_xticks(freqs)
            ax.set_yticks(np.arange(-20, 130, 10))

    # Turn off unused subplots
    for idx in range(num_participants * (2 if split else 1), len(axes.flat)):
        axes.flat[idx].axis('off')

    # Create an overall legend and adjust layout
    fig.legend(['Right Ear', 'Left Ear'], loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    plt.tight_layout()
    plt.show()