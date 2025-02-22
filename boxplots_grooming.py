import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

csv_files = [
    "153353_leg_control_features.csv",
    "161024_leg_ad_features.csv",
    "233914_backLeg_aloe_features.csv"
]
labels = ['Control', 'AD', 'AD with Aloe'] 

def load_data(csv_files, labels):
    all_data = []
    for idx, file in enumerate(csv_files):
        # read the CSV file into a pandas DataFrame
        df = pd.read_csv(file)
        # add a new column to distinguish which video the data belongs to
        df['Video'] = labels[idx]
        # append the DataFrame to the list
        all_data.append(df)
    # concatenate all data into one DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)
    return combined_data

# function to perform t-tests and plot box plots with p-value annotations
def plot_box_plots_with_p_values(combined_data, features):
    group_labels = combined_data['Video'].unique()
    group_positions = {label: idx for idx, label in enumerate(group_labels)}
    
    for feature in features:
        plt.figure(figsize=(10, 6))
        
        # create box plots for the feature across different videos
        sns.boxplot(x='Video', y=feature, data=combined_data, palette='Set2')
        sns.stripplot(x='Video', y=feature, data=combined_data, color='grey', alpha=0.5, jitter=True, dodge=True)

        # perform pairwise t-tests between groups
        max_y = combined_data[feature].max()
        spacing = 0.1 * max_y  # vertical spacing for annotations
        for i, label1 in enumerate(group_labels):
            for j, label2 in enumerate(group_labels):
                if i < j:
                    # get the data for the two groups
                    group1 = combined_data[combined_data['Video'] == label1][feature]
                    group2 = combined_data[combined_data['Video'] == label2][feature]
                    
                    # perform a t-test
                    _, p_val = ttest_ind(group1, group2, equal_var=False)
                    
                    # annotate only the p-value on the plot
                    x1, x2 = group_positions[label1], group_positions[label2]
                    y = max_y + (i + j) * spacing
                    plt.plot([x1, x1, x2, x2], [y, y + spacing / 2, y + spacing / 2, y], lw=1.5, color='black')
                    plt.text((x1 + x2) / 2, y + spacing / 2, f"p = {p_val:.4e}", 
                             ha='center', va='bottom', color='black')

        plt.title(f'{feature} Distribution Across Fly Grooming Videos')
        plt.xlabel('Fly Group')
        plt.ylabel(feature)
        plt.grid(True)
        plt.show()

# load the combined data
combined_data = load_data(csv_files, labels)

features = ["MeanMagnitude", "MaxMagnitude", "StdMagnitude", "MedianMagnitude"]

# generate box plots w/ p-values
plot_box_plots_with_p_values(combined_data, features)