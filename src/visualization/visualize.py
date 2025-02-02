import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Load the processed dataset from a pickle file
df = pd.read_pickle("../../data/interim/01_data_processed.pkl") 

# Extract unique labels (categories) and unique participants from the dataset
labels = df["label"].unique()
participants = df["participant"].unique()

# Iterate through each label (category)
for label in labels:
    # Iterate through each participant
    for participant in participants:
        # Filter the dataset for the specific label and participant
        combined_plot_df = df.query(f"label == '{label}'").query(f"participant == '{participant}'").reset_index()
        
        # Only proceed if there is data for the given label and participant
        if len(combined_plot_df) > 0:
            
            # Create a figure with two vertically stacked subplots sharing the x-axis
            fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
            
            # Plot accelerometer data (x, y, z) on the first subplot
            combined_plot_df[["acc_x", "acc_y", "acc_z"]].plot(ax=ax[0])
            
            # Plot gyroscope data (x, y, z) on the second subplot
            combined_plot_df[["gyr_x", "gyr_y", "gyr_z"]].plot(ax=ax[1])

            # Adjust legend positions for better visibility
            ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, fancybox=True, shadow=True)
            
            # Label the x-axis
            ax[1].set_xlabel("samples")
            
            # Save the figure to the specified directory with a filename based on label and participant
            plt.savefig(f"../../reports/figures/{label.title()} ({participant}).png")
            
            # Display the plot
            plt.show()