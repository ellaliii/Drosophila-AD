import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_curved_graphs_from_csvs(csv_files, video_names, window=10, polynomial_degree=3):
    """
    Plots smoothed direction values (as curved lines using polynomial fits) for multiple videos
    by combining data from multiple CSV files and assigning video names manually

    Args:
        csv_files (list): A list of file paths to CSV files.
        video_names (list): A list of video names corresponding to the CSV files.
        window (int): The size of the rolling window for smoothing.
        polynomial_degree (int): Degree of the polynomial for curve fitting.
    """
    # combine data from all CSV files and assign video names
    all_data = pd.DataFrame()
    for file, video_name in zip(csv_files, video_names):
        df = pd.read_csv(file)
        if "direction_changes" not in df.columns:
            print(f"'direction_changes' column is missing in {file}. Skipping.")
            continue
        df["video_name"] = video_name  # assign video name to a new column
        all_data = pd.concat([all_data, df], ignore_index=True)
    
    plt.figure(figsize=(12, 8))  
    colors = ["#8DD3C7", "#BEBADA", "#FFC107"]  
    
    for i, video_name in enumerate(video_names):
        print(f"\nProcessing video: {video_name}")
        video_data = all_data[all_data["video_name"] == video_name]
        print(f"  - Rows in video_data: {len(video_data)}")
        
        if video_data.empty:
            print(f"  - No data found for video: {video_name}")
            continue
        
        # apply a rolling window to smooth direction values
        smoothed_data = video_data["direction_changes"].rolling(window=min(window, len(video_data))).mean()
        print(f"  - Smoothed data sample:\n{smoothed_data.head(10)}")
        print(f"  - Valid points in smoothed data: {smoothed_data.notna().sum()}")
        
        # generate an implicit time sequence
        time_steps = np.arange(len(smoothed_data))
        
        # polynomial fit to create a smooth curve
        valid_mask = ~smoothed_data.isna()
        if valid_mask.sum() < polynomial_degree + 1:
            print(f"  - Insufficient data for polynomial fitting. Valid points: {valid_mask.sum()}")
            continue
        
        poly_fit = np.polyfit(time_steps[valid_mask], smoothed_data[valid_mask], polynomial_degree)
        print(f"  - Polynomial coefficients for {video_name}: {poly_fit}")
        poly_curve = np.poly1d(poly_fit)
        
        # plot the polynomial curve
        plt.plot(
            time_steps,
            poly_curve(time_steps),
            linestyle="-",
            label=f"Polynomial Fit: {video_name}",
            color=colors[i % len(colors)],
            alpha=0.8
        )
    
    # set title and labels
    plt.title("Curved Direction Distribution Over Time (Polynomial Fit)", fontsize=16)
    plt.xlabel(f"Time Steps (smoothed over {window} samples)", fontsize=14)
    plt.ylabel("Smoothed Direction Changes (pixels/frame)", fontsize=14)
    plt.legend()
    plt.grid()
    plt.show()

csv_files = [
    'fly_features_6119_c.csv', 
    'fly_features_6105-sick.csv', 
    'fly_features_6222_aloe_sick.csv'
]  
video_names = ["6119_c.mov", "6222_aloe_sick.mov", "6105-sick-1.MOV"] 
plot_curved_graphs_from_csvs(csv_files, video_names, window=10, polynomial_degree=3)