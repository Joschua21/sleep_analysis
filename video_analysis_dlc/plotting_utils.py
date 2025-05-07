import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
import os
import colorsys

try:
    import cv2
    import moviepy.editor as mpy
    VIDEO_LIBS_AVAILABLE = True
except ImportError:
    VIDEO_LIBS_AVAILABLE = False
    print("Warning: moviepy or opencv-python not installed. Video generation functionality will be disabled.")
    print("Install them using: pip install moviepy opencv-python")



def interpolate_gaps_conditionally(series: pd.Series, max_gap_length: int) -> pd.Series:
    """
    Interpolates NaN values in a Series using linear interpolation, but only
    if the consecutive NaN gap is shorter than or equal to max_gap_length.

    Args:
        series (pd.Series): The input Series with potential NaNs.
        max_gap_length (int): The maximum length of a NaN gap to interpolate.

    Returns:
        pd.Series: Series with short NaN gaps interpolated.
    """
    s_out = series.copy()
    is_na = s_out.isna()

    if not is_na.any(): # No NaNs, nothing to do
        return s_out

    # Identify groups of consecutive NaNs
    na_group_ids = is_na.ne(is_na.shift()).cumsum()

    # Iterate over each NaN block
    # We are interested in the groups that are True (i.e., are NaN blocks)
    for group_id, na_block_series in s_out[is_na].groupby(na_group_ids[is_na]):
        if not na_block_series.empty and len(na_block_series) <= max_gap_length:
            # This block of NaNs is short enough to interpolate.
            # We perform a full interpolation on the original series (s_out)
            # to get the correct values based on surrounding non-NaNs,
            # and then apply these interpolated values only to the current short gap.
            
            # Temporarily interpolate the whole series to get potential values
            temp_interpolated_series = s_out.interpolate(method='linear', limit_direction='both')
            
            # Copy values from temp_interpolated_series to s_out *only* for this short gap's indices
            s_out.loc[na_block_series.index] = temp_interpolated_series.loc[na_block_series.index]
            
    return s_out

# Add smoothing_window_seconds parameter
def plot_speed(df_dlc, df_displacements, final_bodyparts_list, frame_rate, output_dir, base_filename, plot_individual=True, save_plot=True, smoothing_window_seconds=1.0):
    """
    Plots the average speed and optionally the speed of individual bodyparts over time.
    Can apply a rolling window average to smooth the average speed line.

    Args:
        # ... other args ...
        save_plot (bool, optional): Whether to save the plot to a file. Defaults to True.
        smoothing_window_seconds (float, optional): Duration of the rolling window for smoothing
                                                    the average speed plot in seconds.
                                                    Set to 0 or None to disable smoothing. Defaults to 1.0.
    """
    print(f"\nGenerating speed plot... Individual parts: {plot_individual}, Smoothing: {smoothing_window_seconds}s")
    fig, ax = plt.subplots(figsize=(8, 3))
    # Calculate time axis
    time_seconds = df_dlc.index / frame_rate

    # Plot Average Speed (potentially smoothed)
    if ('analysis', 'speed_pixels_per_second') in df_dlc.columns:
        average_speed = df_dlc[('analysis', 'speed_pixels_per_second')]
        plot_label = 'Average Speed'

        # Apply smoothing if requested
        if smoothing_window_seconds and smoothing_window_seconds > 0:
            # Calculate window size in frames (must be an integer >= 1)
            window_size = max(1, int(smoothing_window_seconds * frame_rate))
            # Apply rolling mean - center=True places the window centered on the point
            # min_periods=1 ensures calculation even if window is not full (e.g., at edges)
            average_speed_smoothed = average_speed.rolling(window=window_size, center=True, min_periods=1).mean()
            plot_data = average_speed_smoothed
            plot_label = f'Average Speed ({smoothing_window_seconds}s Smoothed)'
            print(f"  Applying rolling average with window size: {window_size} frames ({smoothing_window_seconds}s)")
        else:
            plot_data = average_speed # Plot original if no smoothing

        ax.plot(time_seconds, plot_data, label=plot_label, linewidth=2, color='black', zorder=10)
    else:
        print("Warning: Average speed column not found for plotting.")

    # Plot Individual Bodypart Speeds (Optional - not smoothed)
    if plot_individual:
        for bp in final_bodyparts_list:
            displacement_col = f'{bp}_displacement'
            if displacement_col in df_displacements.columns:
                individual_speed = df_displacements[displacement_col] * frame_rate
                ax.plot(time_seconds, individual_speed, label=f'{bp} Speed', alpha=0.6)
            else:
                 print(f"Warning: Displacement column '{displacement_col}' not found for plotting individual speed.")

    # Customize plot
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Speed (pixels/second)')
    ax.set_title('Mouse Speed Over Time')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    # Save plot
    if save_plot:
        # Add suffix if smoothed
        smooth_suffix = f'_smoothed{smoothing_window_seconds}s' if smoothing_window_seconds and smoothing_window_seconds > 0 else ''
        plot_filename = os.path.join(output_dir, base_filename + '_speed_plot' + smooth_suffix + '.png')
        try:
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"Speed plot saved to: {plot_filename}")
        except Exception as e:
            print(f"Error saving plot: {e}")

    return fig


def create_synced_video_with_plot(
    video_path: str,
    speed_data: pd.Series,
    frame_rate: float,
    output_video_path: str,
    median_coords: pd.DataFrame = None, # New parameter for median coordinates
    plot_width_seconds: float = 5.0,
    plot_height_pixels: int = 200,
    median_point_radius: int = 5, # Radius of the median point
    median_point_color: tuple = (0, 0, 0) # Black color for the point (BGR for OpenCV)
):
    if not VIDEO_LIBS_AVAILABLE:
        print("Error: Cannot create video. Required libraries (moviepy, opencv-python) are missing.")
        return

    print(f"Starting synchronized video creation with median point overlay: {output_video_path}")
    original_backend = matplotlib.get_backend()
    print(f"Original matplotlib backend: {original_backend}")
    matplotlib.use('Agg') # Switch to a non-interactive backend for performance
    print(f"Temporarily switched matplotlib backend to: Agg")

    fig = None # Initialize fig to None for the finally block
    video_clip_orig = None

    try:
        video_clip_orig = mpy.VideoFileClip(video_path)
        w, h = video_clip_orig.w, video_clip_orig.h
        vid_duration = video_clip_orig.duration
        vid_fps = video_clip_orig.fps

        if abs(vid_fps - frame_rate) > 1:
            print(f"Warning: Video FPS ({vid_fps}) differs significantly from specified frame_rate ({frame_rate}). Using video FPS.")
            actual_frame_rate = vid_fps
        else:
            actual_frame_rate = frame_rate

        # --- Prepare median coordinates if provided ---
        median_x_np = None
        median_y_np = None
        if median_coords is not None and not median_coords.empty:
            if ('analysis', 'median_x') in median_coords.columns and \
               ('analysis', 'median_y') in median_coords.columns:
                median_x_np = median_coords[('analysis', 'median_x')].to_numpy()
                median_y_np = median_coords[('analysis', 'median_y')].to_numpy()
                print("Median coordinates provided for overlay.")
            else:
                print("Warning: Median coordinates DataFrame provided but 'median_x' or 'median_y' columns are missing. No overlay will be drawn.")
        else:
            print("No median coordinates provided for overlay.")


        # --- Function to draw median point on each frame ---
        def draw_median_on_frame(get_frame, t):
            frame_orig = get_frame(t) # Get the original frame
            frame = frame_orig.copy() # <--- MAKE A WRITABLE COPY
            current_frame_idx = int(t * actual_frame_rate)

            if median_x_np is not None and median_y_np is not None and \
               current_frame_idx < len(median_x_np) and current_frame_idx < len(median_y_np):
                
                mx = median_x_np[current_frame_idx]
                my = median_y_np[current_frame_idx]

                if not np.isnan(mx) and not np.isnan(my):
                    center_coordinates = (int(mx), int(my))
                    # Draw the circle. Note: frame is a NumPy array.
                    # OpenCV uses BGR color format by default.
                    try:
                        cv2.circle(frame, center_coordinates, median_point_radius, median_point_color, -1) # -1 for filled circle
                    except Exception as e_cv2:
                        print(f"Error drawing circle at frame {current_frame_idx} time {t}: {e_cv2}")
            return frame

        # Apply the drawing function to the video clip
        video_clip_processed = video_clip_orig.fl(draw_median_on_frame)
        # --- End Median Point Drawing ---


        speed_data_np = speed_data.fillna(0).to_numpy()
        valid_speeds = speed_data_np[np.isfinite(speed_data_np)]

        if len(valid_speeds) > 0:
            max_s = np.max(valid_speeds) * 1.1 
        else:
            max_s = 0 # Fallback if no valid speed data
        
        if max_s == 0 or np.isnan(max_s) or not np.isfinite(max_s): # Ensure max_speed is a positive finite number
            max_speed_for_plot = 100.0 # Default y-limit if max_s is problematic
        else:
            max_speed_for_plot = max_s

        frames_in_plot_window = int(plot_width_seconds * actual_frame_rate)
        time_per_frame = 1.0 / actual_frame_rate

        fig, ax = plt.subplots(figsize=(w / 80, plot_height_pixels / 80), dpi=80)
        line, = ax.plot([], [], color='r')
        vline = ax.axvline(0, color='lime', linestyle='--', lw=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (px/s)")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim(0, max_speed_for_plot)
        fig.tight_layout(pad=0.5)

        def make_plot_frame(t):
            # ... (rest of make_plot_frame remains the same as your optimized version)
            current_frame = int(t * actual_frame_rate)
            start_frame_idx = max(0, current_frame - frames_in_plot_window)
            end_frame_idx = current_frame + 1
            plot_data_segment = speed_data_np[start_frame_idx:end_frame_idx]
            
            # Ensure time_axis_segment aligns with plot_data_segment
            # It should represent the actual time values for the x-axis of the plot window
            time_axis_plot_window_data = np.arange(start_frame_idx, end_frame_idx) * time_per_frame
            
            line.set_data(time_axis_plot_window_data, plot_data_segment)
            
            # Current time marker position (this 't' is the video's current time)
            vline.set_xdata([t, t])
            
            # Set x-axis limits for the scrolling window effect
            # The window shows `plot_width_seconds` of data, with the current time 't' ideally at the right edge or slightly before.
            plot_window_end_time = t + (time_per_frame * 5) # Show a little bit past current time for context
            plot_window_start_time = max(0.0, plot_window_end_time - plot_width_seconds)

            # Adjust if current time t is less than the plot_width_seconds
            if t < plot_width_seconds - (time_per_frame * 5) : # Ensure the window starts at 0 if t is too small
                plot_window_start_time = 0.0
                plot_window_end_time = plot_width_seconds
            
            ax.set_xlim(plot_window_start_time, plot_window_end_time)
            
            fig.canvas.draw() # This is the expensive call
            img_buf = fig.canvas.buffer_rgba()
            img = np.frombuffer(img_buf, dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            return img[:, :, :3]

        plot_clip = mpy.VideoClip(make_plot_frame, duration=vid_duration)
        plot_clip = plot_clip.resize(newsize=(w, plot_height_pixels))

        # Use the processed video clip (with median point) here
        final_clip = mpy.clips_array([[video_clip_processed], [plot_clip]])

        print(f"Writing final video to {output_video_path}...")
        # ... (write_videofile call remains the same, consider the h264_nvenc option)
        try:
            final_clip.write_videofile(
                output_video_path, fps=actual_frame_rate, codec='h264_nvenc', audio=False,
                threads=4, preset='fast', logger='bar'
            )
            print("Video encoding with h264_nvenc successful.")
        except Exception as e_nvenc:
            print(f"Warning: h264_nvenc encoding failed ({e_nvenc}). Falling back to libx264.")
            final_clip.write_videofile(
                output_video_path, fps=actual_frame_rate, codec='libx264', audio=False,
                threads=4, preset='ultrafast', logger='bar'
            )

        plt.close(fig)
        video_clip_orig.close() # Close the original video clip
        # video_clip_processed doesn't need explicit close if it's just a result of .fl()
        print("Video creation complete.")

    except Exception as e:
        print(f"Error during video creation: {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        if 'video_clip_orig' in locals() and hasattr(video_clip_orig, 'close'): video_clip_orig.close()


# Video conversion for adding sleep bout periods and changes in arousal

def create_synced_video_with_sleep_analysis(
    video_path: str,
    speed_data: pd.Series,
    frame_rate: float,
    output_video_path: str,
    df_sleep_bouts: pd.DataFrame,  # New parameter for sleep bout information
    median_coords: pd.DataFrame = None,
    plot_width_seconds: float = 5.0,
    plot_height_pixels: int = 200,
    median_point_radius: int = 5,
    median_point_color: tuple = (0, 0, 255),  # Red color (BGR for OpenCV)
    sleep_threshold: float = 75.0,  # Speed threshold for sleep
    arousal_low_threshold: float = 30.0,  # Lower threshold for arousal
    arousal_high_threshold: float = 40.0  # Higher threshold for arousal
):
    """
    Create a synchronized video with sleep analysis visualization.
    - Marks sleep periods (speed < sleep_threshold) with light green background
    - Changes line color based on arousal state:
      * Blue during sleep periods (< arousal_low_threshold)
      * Yellow-blue gradient when between arousal_low_threshold and arousal_high_threshold
      * Red-yellow gradient when between arousal_high_threshold and sleep_threshold
      * Black when not in sleep periods
    """
    if not VIDEO_LIBS_AVAILABLE:
        print("Error: Cannot create video. Required libraries (moviepy, opencv-python) are missing.")
        return

    print(f"Starting synchronized video creation with sleep analysis: {output_video_path}")
    original_backend = matplotlib.get_backend()
    print(f"Original matplotlib backend: {original_backend}")
    matplotlib.use('Agg')  # Switch to a non-interactive backend for performance
    print(f"Temporarily switched matplotlib backend to: Agg")

    fig = None  # Initialize fig to None for the finally block
    video_clip_orig = None

    try:
        video_clip_orig = mpy.VideoFileClip(video_path)
        w, h = video_clip_orig.w, video_clip_orig.h
        vid_duration = video_clip_orig.duration
        vid_fps = video_clip_orig.fps

        if abs(vid_fps - frame_rate) > 1:
            print(f"Warning: Video FPS ({vid_fps}) differs significantly from specified frame_rate ({frame_rate}). Using video FPS.")
            actual_frame_rate = vid_fps
        else:
            actual_frame_rate = frame_rate

        # --- Prepare median coordinates if provided ---
        median_x_np = None
        median_y_np = None
        if median_coords is not None and not median_coords.empty:
            if ('analysis', 'median_x') in median_coords.columns and \
               ('analysis', 'median_y') in median_coords.columns:
                median_x_np = median_coords[('analysis', 'median_x')].to_numpy()
                median_y_np = median_coords[('analysis', 'median_y')].to_numpy()
                print("Median coordinates provided for overlay.")
            else:
                print("Warning: Median coordinates DataFrame provided but 'median_x' or 'median_y' columns are missing. No overlay will be drawn.")
        else:
            print("No median coordinates provided for overlay.")

        # --- Prepare sleep bouts data ---
        sleep_periods = []
        if df_sleep_bouts is not None and not df_sleep_bouts.empty:
            for _, bout in df_sleep_bouts.iterrows():
                sleep_periods.append((bout['start_time_s'], bout['end_time_s']))
            print(f"Added {len(sleep_periods)} sleep periods for visualization.")
        else:
            print("No sleep periods data provided.")

        # --- Function to draw median point on each frame ---
        def draw_median_on_frame(get_frame, t):
            frame_orig = get_frame(t)  # Get the original frame
            frame = frame_orig.copy()  # Make a writable copy
            current_frame_idx = int(t * actual_frame_rate)

            if median_x_np is not None and median_y_np is not None and \
               current_frame_idx < len(median_x_np) and current_frame_idx < len(median_y_np):
                
                mx = median_x_np[current_frame_idx]
                my = median_y_np[current_frame_idx]

                if not np.isnan(mx) and not np.isnan(my):
                    center_coordinates = (int(mx), int(my))
                    # Draw the circle
                    try:
                        cv2.circle(frame, center_coordinates, median_point_radius, median_point_color, -1)
                    except Exception as e_cv2:
                        print(f"Error drawing circle at frame {current_frame_idx} time {t}: {e_cv2}")
            return frame

        # Apply the drawing function to the video clip
        video_clip_processed = video_clip_orig.fl(draw_median_on_frame)

        speed_data_np = speed_data.fillna(0).to_numpy()
        valid_speeds = speed_data_np[np.isfinite(speed_data_np)]

        if len(valid_speeds) > 0:
            max_s = np.max(valid_speeds) * 1.1 
        else:
            max_s = 0  # Fallback if no valid speed data
        
        if max_s == 0 or np.isnan(max_s) or not np.isfinite(max_s):
            max_speed_for_plot = 100.0  # Default y-limit if max_s is problematic
        else:
            max_speed_for_plot = max_s

        frames_in_plot_window = int(plot_width_seconds * actual_frame_rate)
        time_per_frame = 1.0 / actual_frame_rate

        fig, ax = plt.subplots(figsize=(w / 80, plot_height_pixels / 80), dpi=80)
        line, = ax.plot([], [], color='k')  # Start with black line
        vline = ax.axvline(0, color='lime', linestyle='--', lw=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speed (px/s)")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_ylim(0, max_speed_for_plot)
        
        # Add threshold lines
        ax.axhline(sleep_threshold, color='r', linestyle='--', lw=0.5, alpha=0.5)
        ax.axhline(arousal_high_threshold, color='orange', linestyle=':', lw=0.5, alpha=0.5)
        ax.axhline(arousal_low_threshold, color='y', linestyle=':', lw=0.5, alpha=0.5)
        
        fig.tight_layout(pad=0.5)

        # Helper function to check if time is in sleep period
        def is_in_sleep_period(t):
            for start, end in sleep_periods:
                if start <= t < end:
                    return True
            return False
        
        # Helper function to determine color based on speed and sleep state
        def get_line_color(t, speed_value):
            if not is_in_sleep_period(t):
                return 'black'
            
            # In sleep period
            if speed_value < arousal_low_threshold:
                return 'blue'  # Base sleep color
            elif speed_value < arousal_high_threshold:
                # Gradient from blue to yellow based on position between thresholds
                ratio = (speed_value - arousal_low_threshold) / (arousal_high_threshold - arousal_low_threshold)
                # Mix blue (0,0,1) and yellow (1,1,0)
                r = ratio
                g = ratio
                b = 1 - ratio
                return (r, g, b)
            elif speed_value < sleep_threshold:
                # Gradient from yellow to red based on position between thresholds
                ratio = (speed_value - arousal_high_threshold) / (sleep_threshold - arousal_high_threshold)
                # Mix yellow (1,1,0) and red (1,0,0)
                r = 1.0
                g = 1.0 - ratio
                b = 0
                return (r, g, b)
            else:
                return 'black'  # Default case
        
        # Spans for sleep periods (created once)
        sleep_spans = []
        for start, end in sleep_periods:
            span = ax.axvspan(start, end, color='palegreen', alpha=0.3, zorder=0)
            sleep_spans.append(span)

        def make_plot_frame(t):
            current_frame = int(t * actual_frame_rate)
            start_frame_idx = max(0, current_frame - frames_in_plot_window)
            end_frame_idx = current_frame + 1
            plot_data_segment = speed_data_np[start_frame_idx:end_frame_idx]
            
            time_axis_plot_window_data = np.arange(start_frame_idx, end_frame_idx) * time_per_frame
            
            # Determine if current time is in sleep period for coloring
            in_sleep = is_in_sleep_period(t)
            
            # Set up plot window limits
            plot_window_end_time = t + (time_per_frame * 5)
            plot_window_start_time = max(0.0, plot_window_end_time - plot_width_seconds)
            
            # Adjust if current time t is less than plot_width_seconds
            if t < plot_width_seconds - (time_per_frame * 5):
                plot_window_start_time = 0.0
                plot_window_end_time = plot_width_seconds
            
            ax.set_xlim(plot_window_start_time, plot_window_end_time)
            
            # Set line data
            line.set_data(time_axis_plot_window_data, plot_data_segment)
            
            # Update line color based on current speed and sleep state
            current_speed = speed_data_np[current_frame] if current_frame < len(speed_data_np) else 0
            line.set_color(get_line_color(t, current_speed))
            
            # Update current time marker
            vline.set_xdata([t, t])
            
            fig.canvas.draw()
            img_buf = fig.canvas.buffer_rgba()
            img = np.frombuffer(img_buf, dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            return img[:, :, :3]

        plot_clip = mpy.VideoClip(make_plot_frame, duration=vid_duration)
        plot_clip = plot_clip.resize(newsize=(w, plot_height_pixels))

        # Use the processed video clip (with median point)
        final_clip = mpy.clips_array([[video_clip_processed], [plot_clip]])

        print(f"Writing final video to {output_video_path}...")
        try:
            final_clip.write_videofile(
                output_video_path, fps=actual_frame_rate, codec='h264_nvenc', audio=False,
                threads=4, preset='fast', logger='bar'
            )
            print("Video encoding with h264_nvenc successful.")
        except Exception as e_nvenc:
            print(f"Warning: h264_nvenc encoding failed ({e_nvenc}). Falling back to libx264.")
            final_clip.write_videofile(
                output_video_path, fps=actual_frame_rate, codec='libx264', audio=False,
                threads=4, preset='ultrafast', logger='bar'
            )

        plt.close(fig)
        video_clip_orig.close()
        print("Video creation with sleep analysis complete.")

    except Exception as e:
        print(f"Error during video creation: {e}")
        import traceback
        traceback.print_exc()
        if 'fig' in locals() and plt.fignum_exists(fig.number):
            plt.close(fig)
        if 'video_clip_orig' in locals() and hasattr(video_clip_orig, 'close'):
            video_clip_orig.close()
    finally:
        matplotlib.use(original_backend)  # Restore the original matplotlib backend
        print(f"Restored matplotlib backend to: {original_backend}")

def plot_body_posture_metric(
    df_dlc,
    metric_column_tuple, # e.g., ('analysis', 'avg_dist_to_median')
    frame_rate,
    output_dir_path,     # Renamed from output_dir for consistency
    base_output_name,    # Renamed from base_filename for consistency
    save_plot=True,
    display_plot=True,
    smoothing_window_seconds_metric=0.25, # Default smoothing for the new metric
    plot_with_speed=True,
    speed_column_tuple=('analysis', 'speed_pixels_per_second'),
    smoothing_window_seconds_speed=0.25  # Default smoothing for speed on this plot
):
    """
    Plots a calculated body posture metric (e.g., average distance of bodyparts to median)
    over time, optionally with speed on a secondary axis.

    Args:
        df_dlc (pd.DataFrame): DataFrame containing the analysis data.
        metric_column_tuple (tuple): Tuple identifying the metric column, e.g., ('analysis', 'avg_dist_to_median').
        frame_rate (float): Video frame rate in FPS.
        output_dir_path (str): Path to the directory where the plot will be saved.
        base_output_name (str): Base name for the output plot file.
        save_plot (bool): Whether to save the plot.
        display_plot (bool): Whether to display the plot.
        smoothing_window_seconds_metric (float): Smoothing window in seconds for the posture metric. 0 for no smoothing.
        plot_with_speed (bool): Whether to plot speed on a secondary y-axis.
        speed_column_tuple (tuple): Tuple identifying the speed column if plotting with speed.
        smoothing_window_seconds_speed (float): Smoothing window in seconds for the speed trace. 0 for no smoothing.
    """
    if metric_column_tuple not in df_dlc.columns:
        print(f"Error: Metric column {metric_column_tuple} not found in DataFrame.")
        if display_plot: # Avoids error if plt is not meant to be shown
             plt.close(plt.gcf()) if plt.get_fignums() else None
        return

    fig, ax1 = plt.subplots(figsize=(15, 6))
    time_axis_seconds = df_dlc.index / frame_rate

    # --- Plot Body Posture Metric ---
    metric_data = df_dlc[metric_column_tuple].copy()
    if smoothing_window_seconds_metric > 0 and frame_rate > 0:
        smoothing_window_frames = int(smoothing_window_seconds_metric * frame_rate)
        if smoothing_window_frames < 1:
            smoothing_window_frames = 1
        metric_data_smoothed = metric_data.rolling(window=smoothing_window_frames, min_periods=1, center=True).mean()
        label_metric = f'Smoothed {metric_column_tuple[-1]} ({smoothing_window_seconds_metric}s window)'
    else:
        metric_data_smoothed = metric_data
        label_metric = f'{metric_column_tuple[-1]} (px)'

    color_metric = 'tab:red'
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel(f'{metric_column_tuple[-1]} (pixels)', color=color_metric)
    ax1.plot(time_axis_seconds, metric_data_smoothed, color=color_metric, label=label_metric, lw=1.5)
    ax1.tick_params(axis='y', labelcolor=color_metric)
    ax1.grid(True, linestyle=':', alpha=0.6)

    lines, labels = ax1.get_legend_handles_labels()

    if plot_with_speed and speed_column_tuple in df_dlc.columns:
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        speed_data = df_dlc[speed_column_tuple].copy()

        if smoothing_window_seconds_speed > 0 and frame_rate > 0:
            smoothing_window_frames_speed = int(smoothing_window_seconds_speed * frame_rate)
            if smoothing_window_frames_speed < 1:
                smoothing_window_frames_speed = 1
            speed_data_smoothed = speed_data.rolling(window=smoothing_window_frames_speed, min_periods=1, center=True).mean()
            label_speed = f'Smoothed Speed ({smoothing_window_seconds_speed}s window)'
        else:
            speed_data_smoothed = speed_data
            label_speed = 'Speed (px/s)'

        color_speed = 'tab:blue'
        ax2.set_ylabel('Speed (pixels/second)', color=color_speed)
        ax2.plot(time_axis_seconds, speed_data_smoothed, color=color_speed, label=label_speed, lw=1, alpha=0.8)
        ax2.tick_params(axis='y', labelcolor=color_speed)
        
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines += lines2
        labels += labels2
    
    fig.suptitle('Body Posture Metric and Speed Over Time', fontsize=14) # Changed title to suptitle
    ax1.legend(lines, labels, loc='upper left')
    fig.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle


    if save_plot:
        plot_filename = os.path.join(output_dir_path, base_output_name + '_body_posture_metric.png')
        try:
            plt.savefig(plot_filename, dpi=300)
            print(f"Body posture metric plot saved to: {plot_filename}")
        except Exception as e:
            print(f"Error saving body posture metric plot: {e}")

    if display_plot:
        plt.show()
    else:
        plt.close(fig)