import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

def plot_direction_changes_comparison(control_file, sick_file, treatment_file):
    # load the datasets
    control_df = pd.read_csv(control_file)
    sick_df = pd.read_csv(sick_file)
    treatment_df = pd.read_csv(treatment_file)

    # add a new column to indicate the group (Control, AD, AD with Aloe)
    control_df['group'] = 'Control'
    sick_df['group'] = 'AD'
    treatment_df['group'] = 'AD with Aloe'

    # concatenate all the dataframes
    combined_df = pd.concat([control_df, sick_df, treatment_df])

    # t-tests
    control_sick_ttest = ttest_ind(control_df['direction_changes'], sick_df['direction_changes'], equal_var=False)
    control_treatment_ttest = ttest_ind(control_df['direction_changes'], treatment_df['direction_changes'], equal_var=False)
    sick_treatment_ttest = ttest_ind(sick_df['direction_changes'], treatment_df['direction_changes'], equal_var=False)

    # store p-values
    p_values = {
        ("Control", "AD"): control_sick_ttest.pvalue,
        ("Control", "AD with Aloe"): control_treatment_ttest.pvalue,
        ("AD", "AD with Aloe"): sick_treatment_ttest.pvalue
    }

    # create a boxplot with individual data points overlaid
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='group', y='direction_changes', data=combined_df, palette="Set3", showfliers=False)
    sns.stripplot(x='group', y='direction_changes', data=combined_df, color='black', alpha=0.5, jitter=True, dodge=True)

    # annotate p-values with staggered heights
    group_positions = {"Control": 0, "AD": 1, "AD with Aloe": 2}
    max_y = combined_df['direction_changes'].max()  # max y-value from the data
    spacing = 0.1 * max_y  # vertical spacing proportional to the data

    for i, ((group1, group2), p_val) in enumerate(p_values.items()):
        x1, x2 = group_positions[group1], group_positions[group2]  # x-coordinates of the groups
        y = max_y + (i + 1) * spacing  # stagger the height based on the index
        h = 0.05 * max_y  # height of the connecting line proportional to max_y
        plt.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=1.5, color='black')  # line connecting groups
        plt.text((x1 + x2) * 0.5, y + h, f"p = {p_val:.4e}", ha='center', va='bottom', color='black')

    # set plot titles and labels
    plt.title('Comparison of Average Direction Changes Across Fly Groups')
    plt.ylabel('Direction Changes')
    plt.xlabel('Fly Group')
    plt.show()

# call the function with the paths to your CSV files
plot_direction_changes_comparison('fly_features_6119_c.csv', 'fly_features_6105-sick.csv', 'fly_features_6222_aloe_sick.csv')