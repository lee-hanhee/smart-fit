import pandas as pd
from glob import glob

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

# Get all CSV files from the MetaMotion data directory
files = glob("../../data/raw/MetaMotion/*.csv")  
data_path = "../../data/raw/MetaMotion/"  # Define the path to the data directory

def read_data_from_files(files):   
    """
    Reads accelerometer and gyroscope data from CSV files, extracts metadata (participant, label, category),
    and processes timestamps into datetime format.
    
    Args:
        files (list): List of file paths to read.

    Returns:
        acc_df (DataFrame): Processed accelerometer data.
        gyr_df (DataFrame): Processed gyroscope data.
    """
    
    # --------------------------------------------------------------
    # Read all files and separate accelerometer and gyroscope data
    # --------------------------------------------------------------
    
    acc_df = pd.DataFrame()  # Initialize empty dataframe for accelerometer data
    gyr_df = pd.DataFrame()  # Initialize empty dataframe for gyroscope data

    # Set number is an identifier for different accelerometer and gyroscope data files
    acc_set = 1  # Initialize accelerometer set counter
    gyr_set = 1  # Initialize gyroscope set counter

    # Loop through all files in the directory
    for f in files:  
        # Extract metadata from the filename
        participant = f.split("-")[0].replace(data_path, "")  # Extract participant ID, e.g., "A"
        label = f.split("-")[1]  # Extract activity label, e.g., "bench"
        category = f.split("-")[2].rstrip("123").rstrip("_MetaWear_2019")  # Extract category (e.g., "heavy" or "light")

        # Read the CSV file into a DataFrame
        df = pd.read_csv(f)  

        # Add extracted metadata as columns
        df["participant"] = participant  
        df["label"] = label  
        df["category"] = category  

        # Identify whether the file contains accelerometer or gyroscope data
        if "Accelerometer" in f:
            df["set"] = acc_set  # Assign a unique set number
            acc_set += 1  # Increment accelerometer set counter
            acc_df = pd.concat([acc_df, df])  # Append to accelerometer dataframe
        
        if "Gyroscope" in f:
            df["set"] = gyr_set  # Assign a unique set number
            gyr_set += 1  # Increment gyroscope set counter
            gyr_df = pd.concat([gyr_df, df])  # Append to gyroscope dataframe
            
    # --------------------------------------------------------------
    # Convert timestamps to datetime and set as index
    # --------------------------------------------------------------
    
    acc_df.index = pd.to_datetime(acc_df["epoch (ms)"], unit="ms")  # Convert milliseconds to datetime
    gyr_df.index = pd.to_datetime(gyr_df["epoch (ms)"], unit="ms")  # Convert milliseconds to datetime

    # Drop unnecessary columns related to time representation
    del acc_df["epoch (ms)"]  
    del acc_df["time (01:00)"]  
    del acc_df["elapsed (s)"]  

    del gyr_df["epoch (ms)"]  
    del gyr_df["time (01:00)"]  
    del gyr_df["elapsed (s)"]  
    
    return acc_df, gyr_df  # Return processed dataframes

# Call the function to process data files
acc_df, gyr_df = read_data_from_files(files)

# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------

# Merge accelerometer and gyroscope data column-wise
data_merged = pd.concat([acc_df.iloc[:, :3], gyr_df], axis=1)  

# Rename columns for clarity
data_merged.columns = [
    "acc_x",  # Accelerometer X-axis
    "acc_y",  # Accelerometer Y-axis
    "acc_z",  # Accelerometer Z-axis
    "gyr_x",  # Gyroscope X-axis
    "gyr_y",  # Gyroscope Y-axis
    "gyr_z",  # Gyroscope Z-axis
    "participant",  # Participant ID
    "label",  # Activity label
    "category",  # Category (e.g., "heavy" or "light")
    "set",  # Set number
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Define the target sampling frequency for resampling
# Accelerometer:    12.500Hz
# Gyroscope:        25.000Hz

sampling = {
    "acc_x": "mean",  
    "acc_y": "mean",  
    "acc_z": "mean",  
    "gyr_x": "mean",  
    "gyr_y": "mean",  
    "gyr_z": "mean",  
    "participant": "last",  # Preserve the last participant ID in each resampling window
    "label": "last",  # Preserve the last activity label
    "category": "last",  # Preserve the last category
    "set": "last",  # Preserve the last set number
}

# Resample the first 1000 rows using 200ms intervals
data_merged[:1000].resample(rule="200ms").apply(sampling)

# --------------------------------------------------------------
# Split data by day and resample each day's data
# --------------------------------------------------------------

# Group data by day and store as a list of DataFrames
days = [g for n, g in data_merged.groupby(pd.Grouper(freq="D"))]

# Resample each day's data to 200ms intervals, drop missing values, and concatenate the results
data_resampled = pd.concat([df.resample(rule="200ms").apply(sampling).dropna() for df in days])

# Ensure the 'set' column remains an integer after resampling
data_resampled["set"] = data_resampled["set"].astype(int)

# Display information about the final processed dataset
data_resampled.info()

# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------

# Save the processed dataset as a pickle file for further analysis
data_resampled.to_pickle("../../data/interim/01_data_processed.pkl")