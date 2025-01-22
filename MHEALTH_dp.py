import pandas as pd
import os
import numpy as np
from scipy.signal import find_peaks, butter, filtfilt
import matplotlib.pyplot as plt

# File path
dataset_path = 'dataset/MHEALTH/'
log_files = [f for f in os.listdir(dataset_path) if f.endswith('.log')]

# Define column names
mhealth_column_names = [
    "chest_acc_X", "chest_acc_Y", "chest_acc_Z",
    "ECG_1", "ECG_2",
    "ankle_acc_X", "ankle_acc_Y", "ankle_acc_Z",
    "ankle_gyro_X", "ankle_gyro_Y", "ankle_gyro_Z",
    "ankle_mag_X", "ankle_mag_Y", "ankle_mag_Z",
    "wrist_acc_X", "wrist_acc_Y", "wrist_acc_Z",
    "wrist_gyro_X", "wrist_gyro_Y", "wrist_gyro_Z",
    "wrist_mag_X", "wrist_mag_Y", "wrist_mag_Z",
    "Activity"
]

# Sampling rate
sampling_rate = 50  # 50 Hz

# Define bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    if low <= 0 or high >= 1:
        raise ValueError("Digital filter critical frequencies must be between 0 and 1.")
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Create a 'clean' folder to save processed files
clean_folder = os.path.join(dataset_path, 'clean')
os.makedirs(clean_folder, exist_ok=True)

# Process each file
for file in log_files:
    file_path = os.path.join(dataset_path, file)
    # Read the data
    data = pd.read_csv(file_path, sep='\t', header=None, names=mhealth_column_names)

    # Create a copy of the ECG signal to process separately
    ecg_signal = data['ECG_1'].copy().values
    activity = data['Activity'].copy().values

    # Apply bandpass filtering to the ECG signal and adjust the filter frequency range
    filtered_ecg = bandpass_filter(ecg_signal, 0.5, 20, sampling_rate, order=2)

    # Detect R peaks and adjust the peak detection strategy
    height_threshold = np.mean(filtered_ecg) + 0.1 * np.std(filtered_ecg)  # Use mean + smaller standard deviation as threshold to relax R-peak detection
    peaks, _ = find_peaks(filtered_ecg, distance=sampling_rate * 0.4, height=height_threshold)

    # Calculate heart rate (bpm)
    if len(peaks) > 1:
        rr_intervals = np.diff(peaks) / sampling_rate  # R-R intervals (in seconds)
        heart_rates = 60 / rr_intervals  # Convert to bpm

        # Create a heart rate time series and interpolate
        heart_rate_series = pd.Series(data=heart_rates, index=peaks[1:])
        heart_rate_interp = heart_rate_series.reindex(range(len(ecg_signal))).interpolate(
            method='linear').ffill().bfill()

        # Replace the ECG column with heart rate data while ensuring only this column is modified
        data.loc[:, 'ECG_1'] = heart_rate_interp
    else:
        data.loc[:, 'ECG_1'] = np.nan  # If no peaks detected, set to NaN

    # Set the ECG_2 column to 0
    data.loc[:, 'ECG_2'] = 0

    # Insert the heart rate column
    data.insert(4, 'Heart_Rate', heart_rate_interp)

    # Remove rows where activity type is 0
    data = data[data['Activity'] != 0]

    # Save the processed file to the 'clean' folder
    save_path = os.path.join(clean_folder, 'clean_' + file)
    data.to_csv(save_path, index=False, sep='\t')
    print(f"Processed and saved: {save_path}")

print("All files have been successfully processed.")
