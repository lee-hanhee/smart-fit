import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------

df = pd.read_pickle("../../data/interim/01_data_processed.pkl")
df = df[df["label"] != "rest"]

acc_r = df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
gyr_r = df["gyr_x"] ** 2 + df["gyr_y"] ** 2 + df["gyr_z"] ** 2
df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------

bench_df = df[df["label"] == "bench"] # Bench df is a data frame with only bench press data
squat_df = df[df["label"] == "squat"] # Squat df is a data frame with only squat data
row_df = df[df["label"] == "row"] # Row df is a data frame with only row data
ohp_df = df[df["label"] == "ohp"] # Ohp df is a data frame with only ohp data
dead_df = df[df["label"] == "dead"] # Dead df is a data frame with only deadlift data

# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------

plot_df = bench_df # Select the data frame to plot
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_x"].plot() # Plot the x-axis acceleration
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_y"].plot() # Plot the y-axis acceleration
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_r"].plot()

plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_x"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_y"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_z"].plot()
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_r"].plot()


# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------

fs = 1000 / 200 # Sampling frequency
LowPass = LowPassFilter() # Create an instance of the LowPassFilter class

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------

bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]] # Select the first set of the bench press data
squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[0]] # Select the first set of the squat data
row_set = row_df[row_df["set"] == row_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"] == ohp_df["set"].unique()[0]]
dead_set = dead_df[dead_df["set"] == dead_df["set"].unique()[0]]

bench_set["acc_r"].plot()

column = "acc_r"
LowPass.low_pass_filter(bench_set, col=column, sampling_frequency=fs, cutoff_frequency=0.4, order=10)[column + "_lowpass"].plot() # Apply the low pass filter to the bench press data

# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------

def count_reps(dataset, cutoff=0.4, order=10, column="acc_r"):
    data = LowPass.low_pass_filter(
        dataset, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order
    )
    indexes = argrelextrema(data[column + "_lowpass"].values, np.greater) # Find local maxima
    peaks = data.iloc[indexes] # Select the local maximas
    
    return len(peaks) # Return the number of repetitions

    fig, ax = plt.subplots()

    plt.plot(dataset[f"{column}_lowpass"])
    plt.plot(peaks[f"{column}_lowpass"], "o", color="red")

    ax.set_ylabel(f"{column}_lowpass")

    exercise = dataset["label"].iloc[0].title()
    category = dataset["category"].iloc[0].title()

    plt.title(f"{category} {exercise}: {len(peaks)} Reps")
    plt.show()

count_reps(bench_set, cutoff=0.4)
count_reps(squat_set, cutoff=0.35)
count_reps(row_set, cutoff=0.65, column="gyr_r")
count_reps(ohp_set, cutoff=0.35)
count_reps(dead_set, cutoff=0.4)

# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------

df["reps"] = df["category"].apply(lambda x: 5 if x == "heavy" else 10) # Create a column with the number of repetitions for each category
rep_df = df.groupby(["label", "category", "set"])["reps"].max().reset_index() # Group by label, category, and set and select the maximum number of repetitions
rep_df["pres_pred"] = 0 # Create a column to store the predicted number of repetitions

for s in df["set"].unique():
    subset = df[df["set"] == s] # Select the subset of the data frame with the same set
    
    column = "acc_r"
    cutoff = 0.4 
    
    if subset["label"].iloc[0] == "row":
        column = "gyr_r"
        cutoff = 0.65
    
    if subset["label"].iloc[0] == "squat":
        cutoff = 0.35
    
    if subset["label"].iloc[0] == "ohp":
        cutoff = 0.35
    
    reps = count_reps(subset, cutoff=cutoff, column=column) # Count the number of repetitions
    
    rep_df.loc[rep_df["set"] == s, "reps_pred"] = reps # Store the predicted number of repetitions

# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------

error = mean_absolute_error(rep_df["reps"], rep_df["reps_pred"]).round(2) # Calculate the mean absolute error