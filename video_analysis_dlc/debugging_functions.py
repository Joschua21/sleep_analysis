import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from collections import Counter

def plot_bodypart_and_median_trajectories( # Consider renaming to plot_coordinate_timelines
    df_dlc, 
    final_bodyparts_list, 
    likelihood_threshold,
    output_dir_path, 
    plot_file_base_name, 
    save_plot=True, 
    display_plot=True
):
    """
    Plots the X and Y coordinates over time (frame number) for selected bodyparts 
    (after likelihood filtering) and their median.

    Each bodypart and the median will have its own subplot showing X(t) and Y(t).
    NaNs in coordinates or points below likelihood threshold will result in gaps.

    Args:
        df_dlc (pd.DataFrame): DataFrame containing DLC data, including raw bodypart
                               coordinates, likelihoods, and pre-calculated 
                               median_x, median_y under the 'analysis' level.
        final_bodyparts_list (list): List of bodypart names to plot.
        likelihood_threshold (float): Likelihood threshold for filtering bodypart coordinates.
        output_dir_path (str): Directory to save the plot.
        plot_file_base_name (str): Base name for the saved plot file.
                                   The plot will be saved as 
                                   f"{plot_file_base_name}_coordinate_timelines.png".
        save_plot (bool): Whether to save the plot.
        display_plot (bool): Whether to display the plot using plt.show().
    """
    num_bodyparts = len(final_bodyparts_list)
    time_axis = df_dlc.index # Use frame numbers as the time axis
    
    if num_bodyparts == 0 and not (('analysis', 'median_x') in df_dlc.columns and ('analysis', 'median_y') in df_dlc.columns):
        print("No bodyparts selected and no median data available for plotting coordinate timelines.")
        return
    
    num_subplots = num_bodyparts + \
                   (1 if (('analysis', 'median_x') in df_dlc.columns and ('analysis', 'median_y') in df_dlc.columns) else 0)

    if num_subplots == 0:
        print("No data to plot for coordinate timelines.")
        return

    fig, axs = plt.subplots(num_subplots, 1, figsize=(12, 2.5 * num_subplots), sharex=True, sharey=False, squeeze=False)
    axs = axs.flatten() 

    plot_idx = 0

    # Plot individual bodypart coordinate timelines
    for bp in final_bodyparts_list:
        ax = axs[plot_idx]
        
        if (bp, 'x') not in df_dlc.columns or \
           (bp, 'y') not in df_dlc.columns or \
           (bp, 'likelihood') not in df_dlc.columns:
            print(f"Warning: Data for bodypart {bp} (x, y, or likelihood) not found. Skipping its timeline plot.")
            ax.set_title(f'Coordinates over Time for: {bp} (Data Missing)')
            ax.text(0.5, 0.5, 'Data Missing', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            plot_idx += 1
            continue

        x_coords_orig = df_dlc[(bp, 'x')]
        y_coords_orig = df_dlc[(bp, 'y')]
        likelihood = df_dlc[(bp, 'likelihood')]
        
        x_coords = x_coords_orig.copy()
        y_coords = y_coords_orig.copy()
        
        mask = likelihood < likelihood_threshold
        x_coords[mask] = np.nan
        y_coords[mask] = np.nan
        
        ax.plot(time_axis, x_coords, label='X-coordinate', linestyle='-', marker=None, markersize=1, alpha=0.7, color='dodgerblue')
        ax.plot(time_axis, y_coords, label='Y-coordinate', linestyle='-', marker=None, markersize=1, alpha=0.7, color='orangered')
        ax.set_title(f'Coordinates over Time for: {bp} (Likelihood > {likelihood_threshold})')
        ax.set_ylabel('Coordinate (pixels)')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='upper right')
        plot_idx += 1

    # Plot median coordinate timelines if data exists
    if ('analysis', 'median_x') in df_dlc.columns and ('analysis', 'median_y') in df_dlc.columns:
        median_ax = axs[plot_idx]
        median_x = df_dlc[('analysis', 'median_x')]
        median_y = df_dlc[('analysis', 'median_y')]

        median_ax.plot(time_axis, median_x, label='Median X', linestyle='-', marker=None, markersize=1, alpha=0.7, color='blue')
        median_ax.plot(time_axis, median_y, label='Median Y', linestyle='-', marker=None, markersize=1, alpha=0.7, color='red')
        median_ax.set_title('Median Point Coordinates over Time (from likelihood-filtered parts)')
        median_ax.set_ylabel('Coordinate (pixels)')
        median_ax.grid(True, linestyle=':', alpha=0.6)
        median_ax.legend(loc='upper right')
        median_ax.set_xlabel('Frame Number') 
        plot_idx +=1
    elif num_bodyparts > 0 and plot_idx > 0 : 
        axs[plot_idx-1].set_xlabel('Frame Number')


    for i in range(plot_idx, len(axs)):
        fig.delaxes(axs[i])
    if plot_idx < len(axs) and plot_idx > 0: 
         fig.set_size_inches(12, 2.5 * plot_idx)


    fig.suptitle('Bodypart and Median Point Coordinates over Time', fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) 

    if save_plot and plot_file_base_name and output_dir_path:
        os.makedirs(output_dir_path, exist_ok=True)
        plot_filename = os.path.join(output_dir_path, f"{plot_file_base_name}_coordinate_timelines.png") # Updated filename
        try:
            plt.savefig(plot_filename, dpi=300)
            print(f"\nCoordinate timelines plot saved to: {plot_filename}")
        except Exception as e:
            print(f"Error saving coordinate timelines plot: {e}")
    elif save_plot:
        print("\nWarning: Could not save coordinate timelines plot. 'plot_file_base_name' or 'output_dir_path' may be missing.")


    if display_plot:
        plt.show()
    else:
        plt.close(fig)


def plot_consecutive_nan_lengths(
    df_dlc,
    final_bodyparts_list,
    likelihood_threshold,
    output_dir_path,
    plot_file_base_name,
    save_plot=True,
    display_plot=True
):
    """
    Plots the distribution of lengths of consecutive NaN intervals for selected bodyparts
    (after likelihood filtering) and their median.

    A frame is considered NaN if either its X or Y coordinate is NaN.

    Args:
        df_dlc (pd.DataFrame): DataFrame containing DLC data.
        final_bodyparts_list (list): List of bodypart names to plot.
        likelihood_threshold (float): Likelihood threshold for filtering bodypart coordinates.
        output_dir_path (str): Directory to save the plot.
        plot_file_base_name (str): Base name for the saved plot file.
        save_plot (bool): Whether to save the plot.
        display_plot (bool): Whether to display the plot.
    """

    def get_consecutive_nan_counts(series_x, series_y):
        """Helper to count lengths of consecutive NaNs in paired series."""
        is_nan_x = pd.isna(series_x)
        is_nan_y = pd.isna(series_y)
        combined_is_nan = is_nan_x | is_nan_y # True if either x or y is NaN

        lengths = []
        current_length = 0
        for val in combined_is_nan:
            if val:
                current_length += 1
            else:
                if current_length > 0:
                    lengths.append(current_length)
                current_length = 0
        if current_length > 0: # Add the last sequence if it ends with NaN
            lengths.append(current_length)
        
        if not lengths:
            return pd.Series(dtype=int)
        return pd.Series(Counter(lengths)).sort_index()

    num_bodyparts = len(final_bodyparts_list)
    
    plot_items = [] # To store (name, nan_counts_series)

    # Process individual bodyparts
    for bp in final_bodyparts_list:
        if (bp, 'x') not in df_dlc.columns or \
           (bp, 'y') not in df_dlc.columns or \
           (bp, 'likelihood') not in df_dlc.columns:
            print(f"Warning: Data for bodypart {bp} not found. Skipping its NaN plot.")
            plot_items.append((f'{bp} (Data Missing)', pd.Series(dtype=int)))
            continue

        x_coords_orig = df_dlc[(bp, 'x')]
        y_coords_orig = df_dlc[(bp, 'y')]
        likelihood = df_dlc[(bp, 'likelihood')]
        
        x_coords = x_coords_orig.copy()
        y_coords = y_coords_orig.copy()
        
        mask = likelihood < likelihood_threshold
        x_coords[mask] = np.nan
        y_coords[mask] = np.nan
        
        nan_counts = get_consecutive_nan_counts(x_coords, y_coords)
        plot_items.append((f'{bp} (Likelihood < {likelihood_threshold})', nan_counts))

    # Process median coordinates
    if ('analysis', 'median_x') in df_dlc.columns and ('analysis', 'median_y') in df_dlc.columns:
        median_x = df_dlc[('analysis', 'median_x')]
        median_y = df_dlc[('analysis', 'median_y')]
        nan_counts_median = get_consecutive_nan_counts(median_x, median_y)
        plot_items.append(('Median Point', nan_counts_median))
    
    if not plot_items:
        print("No data available to plot NaN consecutive lengths.")
        return

    num_subplots = len(plot_items)
    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 2.5 * num_subplots), sharex=False, squeeze=False)
    axs = axs.flatten()

    max_nan_length_overall = 0
    for _, counts_series in plot_items:
        if not counts_series.empty:
            max_nan_length_overall = max(max_nan_length_overall, counts_series.index.max())

    for plot_idx, (title_prefix, nan_counts_series) in enumerate(plot_items):
        ax = axs[plot_idx]
        if not nan_counts_series.empty:
            nan_counts_series.plot(kind='bar', ax=ax, width=0.8, color='skyblue')
            ax.set_title(f'Consecutive NaN Lengths for: {title_prefix}')
            ax.set_ylabel('Frequency of Sequence')
            if max_nan_length_overall > 0 :
                 ax.set_xlim(-0.5, max_nan_length_overall + 0.5) # Adjust x-limit for better bar visibility
        else:
            ax.set_title(f'Consecutive NaN Lengths for: {title_prefix}')
            ax.text(0.5, 0.5, 'No NaN sequences found' if 'Data Missing' not in title_prefix else 'Data Missing', 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            if max_nan_length_overall > 0 :
                ax.set_xlim(-0.5, max_nan_length_overall + 0.5)


        ax.grid(True, linestyle=':', alpha=0.6, axis='y')

    # Set common X label on the last plot
    if num_subplots > 0:
        axs[-1].set_xlabel('Length of Consecutive NaN Sequence (frames)')

    fig.suptitle('Distribution of Consecutive NaN Sequence Lengths', fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])

    if save_plot and plot_file_base_name and output_dir_path:
        os.makedirs(output_dir_path, exist_ok=True)
        plot_filename = os.path.join(output_dir_path, f"{plot_file_base_name}_nan_sequence_lengths.png")
        try:
            plt.savefig(plot_filename, dpi=300)
            print(f"\nNaN sequence lengths plot saved to: {plot_filename}")
        except Exception as e:
            print(f"Error saving NaN sequence lengths plot: {e}")
    elif save_plot:
        print("\nWarning: Could not save NaN sequence lengths plot. 'plot_file_base_name' or 'output_dir_path' may be missing.")

    if display_plot:
        plt.show()
    else:
        plt.close(fig)