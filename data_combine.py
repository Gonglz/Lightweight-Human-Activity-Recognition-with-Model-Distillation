import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import os

# 1. Combine PAMAP2 and mHealth datasets
def combine_datasets(mhealth_path, pamap_path, save_path):
    # Define activity mappings
    mhealth_activity_mapping = {
        1: 'Standing', 2: 'Sitting', 3: 'Lying',
        4: 'Walking', 5: 'Ascending stairs', 9: 'Cycling',
        10: 'Jogging', 11: 'Running', 12: 'Jumping'
    }

    pamap_activity_mapping = {
        1: 'Lying', 2: 'Sitting', 3: 'Standing', 4: 'Walking',
        5: 'Running', 6: 'Cycling', 7: 'Jogging',
        12: 'Ascending stairs', 24: 'Jumping'
    }

    # Define sensor columns to extract
    columns_to_extract = [
        'Activity', 'Heart_Rate',
        'chest_acc_X', 'chest_acc_Y', 'chest_acc_Z',
        'wrist_acc_X', 'wrist_acc_Y', 'wrist_acc_Z',
        'wrist_gyro_X', 'wrist_gyro_Y', 'wrist_gyro_Z',
        'wrist_mag_X', 'wrist_mag_Y', 'wrist_mag_Z',
        'ankle_acc_X', 'ankle_acc_Y', 'ankle_acc_Z',
        'ankle_gyro_X', 'ankle_gyro_Y', 'ankle_gyro_Z',
        'ankle_mag_X', 'ankle_mag_Y', 'ankle_mag_Z'
    ]

    # Load and map mHealth dataset
    def load_and_map_mhealth(file):
        data = pd.read_csv(file, sep='\t', header=0, usecols=columns_to_extract)
        data = data[data['Activity'].isin(mhealth_activity_mapping.keys())]
        data['Activity'] = data['Activity'].map(mhealth_activity_mapping)
        return data

    # Load and map PAMAP dataset
    def load_and_map_pamap(file):
        column_mapping = {
            'activityID': 'Activity',
            'heart_rate': 'Heart_Rate',
            'chest_acc16_X': 'chest_acc_X', 'chest_acc16_Y': 'chest_acc_Y', 'chest_acc16_Z': 'chest_acc_Z',
            'hand_acc16_X': 'wrist_acc_X', 'hand_acc16_Y': 'wrist_acc_Y', 'hand_acc16_Z': 'wrist_acc_Z',
            'hand_gyro_X': 'wrist_gyro_X', 'hand_gyro_Y': 'wrist_gyro_Y', 'hand_gyro_Z': 'wrist_gyro_Z',
            'hand_mag_X': 'wrist_mag_X', 'hand_mag_Y': 'wrist_mag_Y', 'hand_mag_Z': 'wrist_mag_Z',
            'ankle_acc16_X': 'ankle_acc_X', 'ankle_acc16_Y': 'ankle_acc_Y', 'ankle_acc16_Z': 'ankle_acc_Z',
            'ankle_gyro_X': 'ankle_gyro_X', 'ankle_gyro_Y': 'ankle_gyro_Y', 'ankle_gyro_Z': 'ankle_gyro_Z',
            'ankle_mag_X': 'ankle_mag_X', 'ankle_mag_Y': 'ankle_mag_Y', 'ankle_mag_Z': 'ankle_mag_Z'
        }
        data = pd.read_csv(file, sep=',', header=0).rename(columns=column_mapping)
        data = data[data['Activity'].isin(pamap_activity_mapping.keys())]
        data['Activity'] = data['Activity'].map(pamap_activity_mapping)
        return data

    # Load and merge datasets
    mhealth_files = [os.path.join(mhealth_path, f) for f in os.listdir(mhealth_path) if f.endswith('.log')]
    pamap_files = [os.path.join(pamap_path, f) for f in os.listdir(pamap_path) if f.endswith('.csv')]

    combined_data = pd.concat(
        [load_and_map_mhealth(file) for file in mhealth_files] +
        [load_and_map_pamap(file) for file in pamap_files],
        ignore_index=True
    )
    combined_data.to_csv(save_path, index=False)
    print(f"Combined data saved to {save_path}")

# 2. Preprocess data for training
def preprocess_data(input_path, output_path, window_size=100, step_size=20):
    # Load dataset
    combined_data = pd.read_csv(input_path)

    # Separate features and labels
    X = combined_data.drop(columns=['Activity'])
    y = combined_data['Activity']  # Keep labels as a 1D array

    # Handle missing values
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Handle class imbalance
    smote = SMOTE(sampling_strategy='minority', random_state=39)
    under_sampler = RandomUnderSampler(sampling_strategy='majority', random_state=39)
    pipeline = Pipeline([('smote', smote), ('under', under_sampler)])
    X_resampled, y_resampled = pipeline.fit_resample(X_scaled, y)

    # Convert labels to one-hot encoding
    y_resampled = pd.get_dummies(y_resampled).to_numpy()

    # Create time windows
    def create_time_windows(data, labels, window_size, step_size):
        X_windows, y_windows = [], []
        for i in range(0, len(data) - window_size, step_size):
            X_windows.append(data[i:i + window_size])
            y_windows.append(labels[i + window_size - 1])  # Use the last sample label in the window
        return np.array(X_windows), np.array(y_windows)

    X_windows, y_windows = create_time_windows(X_resampled, y_resampled, window_size, step_size)

    # Ensure balanced class distribution in train and validation sets
    def split_by_class(X, y, train_ratio=0.8):
        unique_classes = np.unique(np.argmax(y, axis=1))
        X_train, X_val, y_train, y_val = [], [], [], []
        for cls in unique_classes:
            cls_indices = np.where(np.argmax(y, axis=1) == cls)[0]
            np.random.shuffle(cls_indices)  # Shuffle indices
            split_index = int(len(cls_indices) * train_ratio)
            X_train.extend(X[cls_indices[:split_index]])
            X_val.extend(X[cls_indices[split_index:]])
            y_train.extend(y[cls_indices[:split_index]])
            y_val.extend(y[cls_indices[split_index:]])
        return np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)

    X_train, X_val, y_train, y_val = split_by_class(X_windows, y_windows, train_ratio=0.8)

    # Save preprocessed data
    np.savez(output_path + 'preprocessed_train_test.npz', X_train=X_train, X_test=X_val, y_train=y_train, y_test=y_val)
    print(f"Preprocessing completed. Data saved to {output_path}preprocessed_train_test.npz")
    print(f"Data shapes -> X_train: {X_train.shape}, X_val: {X_val.shape}, y_train: {y_train.shape}, y_val: {y_val.shape}")


# Main execution
if __name__ == "__main__":
    # Step 1: Combine datasets
    combine_datasets('dataset/MHEALTH/clean/', 'dataset/PAMAP2/Processed/clean/', 'dataset/combined_har_data.csv')

    # Step 2: Preprocess data
    preprocess_data('dataset/combined_har_data.csv', 'dataset/', window_size=100, step_size=20)
