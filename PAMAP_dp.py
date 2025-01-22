import pandas as pd
import os
import matplotlib.pyplot as plt

# Define column names
pamap2_column_names = [
    "timestamp", "activityID", "heart_rate",
    "hand_temp", "hand_acc16_X", "hand_acc16_Y", "hand_acc16_Z", "hand_acc6_X", "hand_acc6_Y", "hand_acc6_Z",
    "hand_gyro_X", "hand_gyro_Y", "hand_gyro_Z", "hand_mag_X", "hand_mag_Y", "hand_mag_Z",
    "hand_ori_1", "hand_ori_2", "hand_ori_3", "hand_ori_4",
    "chest_temp", "chest_acc16_X", "chest_acc16_Y", "chest_acc16_Z", "chest_acc6_X", "chest_acc6_Y", "chest_acc6_Z",
    "chest_gyro_X", "chest_gyro_Y", "chest_gyro_Z", "chest_mag_X", "chest_mag_Y", "chest_mag_Z",
    "chest_ori_1", "chest_ori_2", "chest_ori_3", "chest_ori_4",
    "ankle_temp", "ankle_acc16_X", "ankle_acc16_Y", "ankle_acc16_Z", "ankle_acc6_X", "ankle_acc6_Y", "ankle_acc6_Z",
    "ankle_gyro_X", "ankle_gyro_Y", "ankle_gyro_Z", "ankle_mag_X", "ankle_mag_Y", "ankle_mag_Z",
    "ankle_ori_1", "ankle_ori_2", "ankle_ori_3", "ankle_ori_4"
]

# File paths
protocol_path = 'dataset/PAMAP2/Protocol/'  # Path to Protocol data files
optional_path = 'dataset/PAMAP2/Optional/'  # Path to Optional data files
input_directory = 'dataset/PAMAP2/Processed/'  # Directory to store preprocessed data
output_directory = 'dataset/PAMAP2/Processed/clean/'  # Directory to store filtered data

# Create directories
os.makedirs(input_directory, exist_ok=True)
os.makedirs(output_directory, exist_ok=True)

# Define file lists
protocol_files = [f"subject{str(i).zfill(3)}.dat" for i in range(101, 110)]
optional_files = [f"subject{str(i).zfill(3)}.dat" for i in [101, 105, 106, 108, 109]]

# Combine Protocol and Optional files and label their sources
all_files = [(os.path.join(protocol_path, file), 'protocol') for file in protocol_files] + \
            [(os.path.join(optional_path, file), 'optional') for file in optional_files]

# **Step 1: Interpolate missing heart rate values**
for file_path, source_type in all_files:
    try:
        # Load data
        pamap2_data = pd.read_csv(file_path, sep=' ', header=None, names=pamap2_column_names)

        # Interpolate missing heart rate values
        pamap2_data['heart_rate'] = pamap2_data['heart_rate'].interpolate(method='linear').ffill().bfill()

        # Save interpolated data
        prefix = source_type + "_"
        output_file_name = prefix + os.path.basename(file_path).replace('.dat', '_interpolated.csv')
        output_file_path = os.path.join(input_directory, output_file_name)
        pamap2_data.to_csv(output_file_path, index=False)

        print(f"Interpolated data saved to {output_file_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# (Optional) Visualize interpolation results for a specific file
file_to_visualize = os.path.join(input_directory, "protocol_subject101_interpolated.csv")
if os.path.exists(file_to_visualize):
    data_to_visualize = pd.read_csv(file_to_visualize)
    plt.figure(figsize=(12, 6))
    plt.plot(data_to_visualize['timestamp'], data_to_visualize['heart_rate'], label='Interpolated Heart Rate')
    plt.xlabel('Timestamp (seconds)')
    plt.ylabel('Heart Rate (bpm)')
    plt.title('Heart Rate Over Time (Protocol Subject 101)')
    plt.legend()
    plt.show()

# **Step 2: Filter rows with activityID = 0**
csv_files = [f for f in os.listdir(input_directory) if f.endswith('.csv')]

for file_name in csv_files:
    input_file_path = os.path.join(input_directory, file_name)
    output_file_path = os.path.join(output_directory, file_name)

    # Load CSV file
    df = pd.read_csv(input_file_path)

    # Filter out rows where activityID is 0
    df_filtered = df[df['activityID'] != 0]

    # Save filtered data
    df_filtered.to_csv(output_file_path, index=False)
    print(f"Filtered data saved to {output_file_path}")

print("All files have been successfully processed and filtered.")
