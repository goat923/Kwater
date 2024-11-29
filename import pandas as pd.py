import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Set display options to show all columns
pd.set_option('display.max_columns', None)

# Load the training dataset
train_data = pd.read_csv("train/TRAIN_B.csv")

# Convert the timestamp column to a datetime object for easier time-based analysis
# train_data['timestamp'] = pd.to_datetime(train_data['timestamp'])
train_data['timestamp'] = pd.to_datetime(train_data['timestamp'], format='%y/%m/%d %H:%M')

# Preview the dataset
print("Dataset Overview")
print(train_data.head())
print("\nSummary Statistics")
print(train_data.describe())

# Check for missing values
print("\nMissing Values")
print(train_data.isnull().sum())
if train_data.isnull().sum().sum() == 0:
    print("No missing values found.")

# Check if 'anomaly' is flagged correctly
# Sum all P1_flag to P26_flag columns and compare to 'anomaly'
sensor_flags = train_data.filter(regex='P\d+_flag').astype(bool)
train_data['sensor_flags_sum'] = sensor_flags.sum(axis=1).astype(bool)
anomaly_flag_validation = (train_data['anomaly'] == 1) & (train_data['sensor_flags_sum'] == 0)

# Print validation results
if anomaly_flag_validation.any():
    print("Warning: Some anomalies are flagged incorrectly (anomaly=1, but no sensor flags are set).")
else:
    print("Validation Passed: 'anomaly' flag is set correctly based on sensor flags.")

# Drop the helper column after validation
train_data.drop('sensor_flags_sum', axis=1, inplace=True)