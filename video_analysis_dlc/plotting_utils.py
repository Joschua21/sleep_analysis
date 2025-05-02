import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

try:
    import cv2
    import moviepy.editor as mpy
    from moviepy.video.io.bindings import mplfig_to_npimage
    VIDEO_LIBS_AVAILABLE = True
except ImportError:
    VIDEO_LIBS_AVAILABLE = False
    print("Warning: moviepy or opencv-python not installed. Video generation functionality will be disabled.")
    print("Install them using: pip install moviepy opencv-python")


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

    # Calculate time axis
    time_seconds = df_dlc.index / frame_rate

    fig, ax = plt.subplots(figsize=(15, 6))

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

    # Explicitly display the plot
    plt.show()

    # Close the figure
    plt.close(fig)




def create_synced_video_with_plot(
    video_path: str,
    speed_data: pd.Series, # The smoothed speed data (pixels/sec)
    frame_rate: float,
    output_video_path: str,
    plot_width_seconds: float = 5.0, # How many seconds the plot window should show
    plot_height_pixels: int = 200, # Height of the plot area
):
    """
    Creates a video with the original video on top and a synchronized,
    scrolling speed plot below.

    Args:
        video_path: Path to the original input video file.
        speed_data: Pandas Series containing the smoothed speed data, indexed by frame number.
        frame_rate: Frame rate of the video.
        output_video_path: Path to save the combined output video.
        plot_width_seconds: Duration (in seconds) the scrolling plot window should display.
        plot_height_pixels: Height (in pixels) for the plot area.
        smoothing_window_frames: The window size used to smooth the speed data.
    """
    if not VIDEO_LIBS_AVAILABLE:
        print("Error: Cannot create video. Required libraries (moviepy, opencv-python) are missing.")
        return

    print(f"Starting synchronized video creation: {output_video_path}")
    # --- Implementation Details ---
    # 1. Load the video clip using moviepy.VideoFileClip(video_path)
    # 2. Get video properties: duration, width, height, fps. Ensure fps matches frame_rate.
    # 3. Prepare speed data: Ensure it's aligned with frame numbers. Handle NaNs (e.g., fill with 0).
    # 4. Calculate plot parameters:
    #    - Plot width should match video width.
    #    - Determine max speed for y-axis limit (add some padding).
    #    - Calculate how many frames correspond to plot_width_seconds.
    # 5. Define a function `make_plot_frame(t)`:
    #    - This function is called by moviepy for each frame time `t`.
    #    - Calculate the current frame index: `current_frame = int(t * frame_rate)`
    #    - Determine the range of frames to display in the plot:
    #      - `start_frame = max(0, current_frame - frames_in_plot_window)`
    #      - `end_frame = current_frame`
    #    - Select the speed data for this window: `speed_segment = speed_data[start_frame:end_frame]`
    #    - Create a matplotlib figure and axes:
    #      - Set x-limits from `start_frame / frame_rate` to `end_frame / frame_rate` (time in seconds).
    #      - Set y-limits from 0 to max_speed.
    #      - Plot the `speed_segment` against its corresponding time values.
    #      - Add a vertical line or marker at the current time `t`.
    #      - Customize plot appearance (labels, title, line color).
    #    - Convert the matplotlib figure to a NumPy array image using `mplfig_to_npimage(fig)`.
    #    - Close the figure (`plt.close(fig)`) to prevent memory leaks.
    #    - Return the NumPy array image.
    # 6. Create the plot clip: `plot_clip = mpy.VideoClip(make_plot_frame, duration=video_clip.duration)`
    # 7. Resize the plot clip image width to match the video width, maintaining aspect ratio or using the specified plot_height_pixels. Use `plot_clip.resize(width=video_clip.width, height=plot_height_pixels)`.
    # 8. Combine the clips vertically: `final_clip = mpy.clips_array([[video_clip], [plot_clip]])`
    # 9. Write the final video: `final_clip.write_videofile(output_video_path, fps=frame_rate, codec='libx264')` # Choose appropriate codec
    # 10. Close clips to release resources.
    # --- End Implementation Details ---

    # Placeholder implementation (replace with actual moviepy code)
    try:
        # Load video
        video_clip = mpy.VideoFileClip(video_path)
        w, h = video_clip.w, video_clip.h
        vid_duration = video_clip.duration
        vid_fps = video_clip.fps
        if abs(vid_fps - frame_rate) > 1: # Allow minor difference
             print(f"Warning: Video FPS ({vid_fps}) differs significantly from specified frame_rate ({frame_rate}). Using video FPS.")
             actual_frame_rate = vid_fps
        else:
             actual_frame_rate = frame_rate # Use the one from analysis

        # Prepare data
        speed_data_np = speed_data.fillna(0).to_numpy() # Fill NaNs for plotting
        max_speed = np.nanmax(speed_data_np) * 1.1 # Add 10% padding
        if max_speed == 0: max_speed = 100 # Avoid zero limit

        # Plot parameters
        frames_in_plot_window = int(plot_width_seconds * actual_frame_rate)
        time_per_frame = 1.0 / actual_frame_rate

        # Matplotlib figure for reuse
        fig, ax = plt.subplots(figsize=(w / 80, plot_height_pixels / 80), dpi=80) # Adjust figsize based on desired output pixels

        # Function to generate plot frame
        def make_plot_frame(t):
            current_frame = int(t * actual_frame_rate)
            # Data fetching window remains the same (up to current time)
            start_frame_idx = max(0, current_frame - frames_in_plot_window)
            end_frame_idx = current_frame + 1 # Include current frame

            # Data segment for the plot window
            plot_data_segment = speed_data_np[start_frame_idx:end_frame_idx]
            # Time axis for the data points being plotted
            time_axis = np.arange(start_frame_idx, end_frame_idx) * time_per_frame

            # Clear previous plot and redraw
            ax.clear()
            ax.plot(time_axis, plot_data_segment, color='r') # Plot data up to time t
            ax.axvline(t, color='lime', linestyle='--', lw=1) # Line for current time t

            # --- Adjust X-axis limits for the view ---
            # Define the total width of the view (plot_width_seconds + buffer)
            view_width_seconds = plot_width_seconds + 1.0
            # Calculate the end time for the x-axis view (current time + buffer)
            plot_end_time = t + 1.0
            # Calculate the start time for the x-axis view
            plot_start_time = max(0.0, plot_end_time - view_width_seconds)
            # Special handling for the beginning: ensure the window width is maintained
            if t < plot_width_seconds:
                 plot_end_time = view_width_seconds # Keep the window fixed until t > plot_width_seconds

            ax.set_xlim(plot_start_time, plot_end_time)
            # --- End Adjust X-axis limits ---

            ax.set_ylim(0, max_speed)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Speed (px/s)")
            ax.grid(True, linestyle=':', alpha=0.6)
            fig.tight_layout(pad=0.5) # Adjust padding

            fig.canvas.draw()
            # Get the RGBA buffer from the figure
            img_buf = fig.canvas.buffer_rgba()
            # Convert the buffer to a numpy array
            img = np.frombuffer(img_buf, dtype=np.uint8)
            # Reshape the array to (height, width, 4)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # Return the image array (moviepy usually handles RGB or RGBA)
            return img[:, :, :3] # Return RGB
        # Create the plot video clip
        plot_clip = mpy.VideoClip(make_plot_frame, duration=vid_duration)

        # Ensure plot clip has correct dimensions before combining
        plot_clip = plot_clip.resize(newsize=(w, plot_height_pixels))

        # Combine video and plot
        final_clip = mpy.clips_array([[video_clip], [plot_clip]])

        # Write the output video
        print(f"Writing final video to {output_video_path}...")
        final_clip.write_videofile(output_video_path, fps=actual_frame_rate, codec='libx264', audio=False, threads=4, logger='bar') # Suppress audio, use multiple threads

        # Close resources
        plt.close(fig)
        video_clip.close()
        plot_clip.close()
        final_clip.close()
        print("Video creation complete.")

    except Exception as e:
        print(f"Error during video creation: {e}")
        # Ensure figure is closed if error occurs
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig)
        # Close clips if they were opened
        if 'video_clip' in locals(): video_clip.close()
        if 'plot_clip' in locals(): plot_clip.close()
        if 'final_clip' in locals(): final_clip.close()