import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("fly_f3.csv")

def plot_smoothed_direction_for_multiple_videos(video_names, window=10):
    """
    plots smoothed speed values for multiple videos on the same graph

    Args:
        video_names (list): A list of video names to filter data for.
        window (int): The size of the rolling window for smoothing.
    """
    plt.figure(figsize=(12, 8))  
    
    colors = ["#8DD3C7", "#FFFFB3", "#BEBADA"]  
    
    for i, video_name in enumerate(video_names):
        # filter the data for the specific video_name
        video_data = df[df["video_name"] == video_name]
        
        # apply a rolling window to smooth speed values
        smoothed_data = video_data["speed"].rolling(window=window).mean()
        
        # generate an implicit time sequence
        time_steps = range(len(smoothed_data))
        
        # plot the smoothed speed values for the current video
        plt.plot(time_steps, smoothed_data, label=f"Smoothed Speed for {video_name}", color=colors[i % len(colors)])
    
    # set title and labels
    plt.title("Smoothed Speed Distribution Over Time")
    plt.xlabel(f"Time Steps (smoothed over {window} samples)")
    plt.ylabel("Smoothed Speed (pixels/frame)")
    plt.ylim(0, 200)  
    plt.legend()
    plt.grid()
    plt.show()

# example usage
video_names = ["6119_c.mov", "6222_aloe_sick.mov", "6105-sick-1.MOV"]  # List of three video names
plot_smoothed_direction_for_multiple_videos(video_names, window=10)