import sys
import os
import json
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler

# Ensure the correct path is added (modify this as needed for your system)
sys.path.append(os.path.abspath('\\media\\mohak\\DiskG\\nuscenes'))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SynchronizedDataset(Dataset):
    def __init__(self, imu_data, can_bus_pose_data, window_size, window_shift):
        if window_shift is None:
            window_shift = window_size
        self.imu_data = imu_data
        self.can_bus_pose_data = can_bus_pose_data
        self.synchronized_data = self.synchronize_data()
        self.window_size = window_size
        self.start_indices = list(range(0, len(self.synchronized_data) - window_size + 1, window_shift))
        logging.info(f"Number of windows: {len(self.start_indices)} (generated from {len(self.synchronized_data)} samples with window size {window_size} and shift {window_shift})")

    def __len__(self):
        return len(self.start_indices)

    def __getitem__(self, idx):
        start_index = self.start_indices[idx]
        window_indices = list(range(start_index, start_index + self.window_size))
        window_data = [self.synchronized_data[i] for i in window_indices]
        
        imu_window = [sample['imu'] for sample in window_data]
        pose_window = [sample['pose'] for sample in window_data]

        imu_window = {
            'data': torch.tensor([sample['linear_accel'] + sample['rotation_rate'] for sample in imu_window], dtype=torch.float32)
        }
        pose_window = {
            'pose_and_orientation': torch.tensor(
                [sample['pos'] + sample['orientation'] + sample['velocity'] for sample in pose_window], dtype=torch.float32
            )
        }
        sample = {
            'imu': imu_window,
            'pose': pose_window
        }
        return sample

    def synchronize_data(self):
        synchronized_data = []
        imu_idx = 0
        origin = None

        linear_accel_scaler = StandardScaler()
        rotation_rate_scaler = StandardScaler()
        pos_scaler = StandardScaler()
        orientation_scaler = StandardScaler()
        velocity_scaler = StandardScaler()

        # Extract data for normalization
        linear_accel_data = [sample['linear_accel'] for sample in self.imu_data]
        rotation_rate_data = [sample['rotation_rate'] for sample in self.imu_data]
        pos_data = [sample['pos'] for sample in self.can_bus_pose_data]
        orientation_data = [sample['orientation'] for sample in self.can_bus_pose_data]
        velocity_data = [sample['vel'] for sample in self.can_bus_pose_data]

        # Fit scalers
        linear_accel_scaler.fit(linear_accel_data)
        rotation_rate_scaler.fit(rotation_rate_data)
        pos_scaler.fit(pos_data)
        orientation_scaler.fit(orientation_data)
        velocity_scaler.fit(velocity_data)

        for can_sample in self.can_bus_pose_data:
            can_timestamp = can_sample['utime']

            # Find the closest IMU sample to the CAN bus sample
            while imu_idx < len(self.imu_data) - 1 and self.imu_data[imu_idx + 1]['utime'] < can_timestamp:
                imu_idx += 1

            imu_sample = self.imu_data[imu_idx]

            if origin is None:
                origin = can_sample['pos']

            # Calculate relative position to the origin
            #relative_pos = [p - o for p, o in zip(can_sample['pos'], origin)]

            # Normalize the data
            normalized_linear_accel = linear_accel_scaler.transform([imu_sample['linear_accel']])[0]
            normalized_rotation_rate = rotation_rate_scaler.transform([imu_sample['rotation_rate']])[0]
            normalized_pos = pos_scaler.transform([can_sample['pos']])[0]
            normalized_orientation = orientation_scaler.transform([can_sample['orientation']])[0]
            normalized_velocity = velocity_scaler.transform([can_sample['vel']])[0]

            # Extract relevant data from IMU and CAN bus samples and create synchronized sample
            synchronized_sample = {
                'utime': imu_sample['utime'],
                'imu': {
                    'linear_accel': normalized_linear_accel.tolist(),
                    'rotation_rate': normalized_rotation_rate.tolist()
                },
                'pose': {
                    'pos': normalized_pos.tolist(),
                    'orientation': normalized_orientation.tolist(),
                    'velocity': normalized_velocity.tolist()
                }
            }
            synchronized_data.append(synchronized_sample)

        logging.info(f"Synchronized {len(synchronized_data)} samples")
        return synchronized_data

# Load data
# imu_file = '/home/mohak/Thesis/BEVFormer/data/nuscenes/can_bus/scene-0001_ms_imu.json'
# pose_file = '/home/mohak/Thesis/BEVFormer/data/nuscenes/can_bus/scene-0001_pose.json'

# with open(imu_file, 'r') as f:
#     imu_data = json.load(f)

# with open(pose_file, 'r') as f:
#     can_bus_pose_data = json.load(f)

# # Initialize dataset
# window_size = 10
# window_shift = None
# dataset = SynchronizedDataset(imu_data, can_bus_pose_data, window_size, window_shift)

# # Create a DataLoader to iterate through the dataset
# dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

# # Fetch a few samples and verify their content
# for i, sample in enumerate(dataloader):
#     imu_window = sample['imu']
#     pose_window = sample['pose']

#     # Check the shape of the imu and pose windows
#     assert imu_window['data'].shape[1] == window_size, f"Sample {i} IMU linear_accel window size incorrect: {imu_window['data'].shape[1]}"
#     assert imu_window['data'].shape[2] == 6, f"Sample {i} IMU rotation_rate window size incorrect: {imu_window['data'].shape[2]}"
#     assert pose_window['pose_and_orientation'].shape[1] == window_size, f"Sample {i} Pose window size incorrect: {pose_window['pose_and_orientation'].shape[1]}"
#     assert pose_window['pose_and_orientation'].shape[2] == 10, f"Sample {i} Pose orientation window size incorrect: {pose_window['pose_and_orientation'].shape[2]}"

#     # Print the first sample to visually verify the data
#     if i == 0:
#         print(f"First sample IMU window: {imu_window}")
#         print(f"First sample Pose window: {pose_window}")
    
#     # Stop after a few samples for the test
#     if i >= 5:
#         break

# print("All tests passed!")
