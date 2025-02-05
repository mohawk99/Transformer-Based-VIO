import sys
import os
import json
import logging
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch.nn.functional as F



# Ensure the correct path is added (modify this as needed for your system)
sys.path.append(os.path.abspath('\\media\\mohak\\DiskG\\nuscenes'))


def load_and_pair_files(folder_path):
    """
    Load and pair IMU and pose JSON files scene by scene.

    Args:
        folder_path (str): Path to the folder containing IMU and pose JSON files.

    Returns:
        dict: Dictionary with scene identifiers as keys and paired IMU/pose data as values.
    """
    imu_files = [f for f in os.listdir(folder_path) if "_ms_imu.json" in f]
    pose_files = [f for f in os.listdir(folder_path) if "_pose.json" in f]

    paired_files_by_scene = {}
    for imu_file in imu_files:
        scene_id = imu_file.replace("_ms_imu.json", "")
        pose_file = f"{scene_id}_pose.json"
        if pose_file in pose_files:
            with open(os.path.join(folder_path, imu_file), 'r') as imu_f:
                imu_data = json.load(imu_f)
            with open(os.path.join(folder_path, pose_file), 'r') as pose_f:
                pose_data = json.load(pose_f)
            paired_files_by_scene[scene_id] = (imu_data, pose_data)
        else:
            logging.warning(f"No matching pose file for IMU file: {imu_file}")

    if not paired_files_by_scene:
        raise ValueError("No paired IMU and pose files found.")
    logging.info(f"Found paired data for {len(paired_files_by_scene)} scenes.")
    return paired_files_by_scene


def create_scene_datasets(paired_files_by_scene, window_size, window_shift):
    """
    Prepare scene-specific datasets with normalization and windowing.

    Args:
        paired_files_by_scene (dict): Dictionary with scene identifiers as keys and paired IMU/pose data as values.
        window_size (int): Size of the data window.
        window_shift (int): Shift between consecutive windows.

    Returns:
        list: A list of synchronized windows for all scenes.
    """
    all_windows = []

    for scene_id, (imu_data, pose_data) in paired_files_by_scene.items():
        # Calculate min and max translation values
        translations = [sample['pos'] for sample in pose_data]
        translations = list(zip(*translations))  # Transpose to get x, y, z as separate lists
        min_translation = [min(dim) for dim in translations]
        max_translation = [max(dim) for dim in translations]

        # Handle zero range for normalization
        range_translation = [
            max_translation[i] - min_translation[i] if max_translation[i] != min_translation[i] else 1.0
            for i in range(3)
        ]

        # Normalize translations to the range (0, 1)
        normalized_translations = [
            [
                (pose[0] - min_translation[0]) / range_translation[0],
                (pose[1] - min_translation[1]) / range_translation[1],
                (pose[2] - min_translation[2]) / range_translation[2] if range_translation[2] != 1.0 else 0.5,
            ]
            for pose in [sample['pos'] for sample in pose_data]
        ]

        # Update pose data with normalized translations
        normalized_pose_data = [
            {
                'pos': normalized_translations[i],
                'orientation': sample['orientation'],
                'vel': sample['vel'],
                'utime': sample['utime'],
                'accel' : sample['accel'],
                'rotation_rate' : sample['rotation_rate']
            }
            for i, sample in enumerate(pose_data)
        ]

        # Synchronize and windowing
        synchronized_data = synchronize_scene_data(imu_data, normalized_pose_data)
        scene_windows = create_windows(synchronized_data, window_size, window_shift)
        all_windows.extend(scene_windows)

    logging.info(f"Generated {len(all_windows)} windows across all scenes.")
    return all_windows





def synchronize_scene_data(imu_data, pose_data):
    """
    Synchronize IMU and pose data for a single scene.

    Args:
        imu_data (list): Normalized IMU data for a scene.
        pose_data (list): Normalized pose data for a scene.

    Returns:
        list: Synchronized data for the scene.
    """
    synchronized_data = []
    imu_idx = 0

    for pose_sample in pose_data:
        pose_timestamp = pose_sample['utime']
        while imu_idx < len(imu_data) - 1 and imu_data[imu_idx + 1]['utime'] < pose_timestamp:
            imu_idx += 1

        imu_sample = imu_data[imu_idx]
        synchronized_data.append({'imu': imu_sample, 'pose': pose_sample})

    return synchronized_data


def create_windows(synchronized_data, window_size=200, window_shift=10):
    windows = []
    target_start = window_size//2 - window_shift//2  # frame 95
    target_end = window_size//2 + window_shift//2    # frame 105
    
    for start_idx in range(0, len(synchronized_data) - window_size + 1, window_shift):
        window = synchronized_data[start_idx:start_idx + window_size]
        
        # Get IMU data for full window
        # imu_window = {
        #     'acc': torch.tensor([sample['imu']['linear_accel'] for sample in window], dtype=torch.float32),
        #     'gyro': torch.tensor([sample['imu']['rotation_rate'] for sample in window], dtype=torch.float32)
        # }
        
        #Test with data in pose
        imu_window = {
            'acc': torch.tensor([sample['pose']['accel'] for sample in window], dtype=torch.float32),
            'gyro': torch.tensor([sample['pose']['rotation_rate'] for sample in window], dtype=torch.float32)
        }

        # Get target frames (95 and 105)
        frame_95 = window[target_start]
        frame_105 = window[target_end]
        
        # Get positions and orientations as numpy arrays
        p_a = np.array(frame_95['pose']['pos'], dtype=np.float32)
        p_b = np.array(frame_105['pose']['pos'], dtype=np.float32)
        q_a = np.array(frame_95['pose']['orientation'], dtype=np.float32)
        q_b = np.array(frame_105['pose']['orientation'], dtype=np.float32)
        
        # Calculate relative position in body frame using numpy
        R_a = quaternion_to_rotation_matrix_numpy(q_a)
        #delta_p = np.matmul(R_a.T, (p_b - p_a))

        #Assuming same frame
        delta_p = p_b - p_a 
        
        # Convert to torch tensors for quaternion operations
        q_a_torch = torch.from_numpy(q_a)
        q_b_torch = torch.from_numpy(q_b)
        
        # Calculate relative orientation
        delta_q = quaternion_multiply(quaternion_conjugate(q_a_torch), q_b_torch)
        
        # Convert position to tensor and combine
        delta_p_torch = torch.from_numpy(delta_p)
        target_pose = torch.cat([delta_p_torch, delta_q])
        
        pose_window = {
            'pose_and_orientation': target_pose
        }
        
        windows.append({'imu': imu_window, 'pose': pose_window})
    
    return windows



# Utility functions
def quaternion_multiply(q1, q2):
    # Implement quaternion multiplication
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    return torch.stack([w, x, y, z], dim=-1)

def quaternion_conjugate(q):
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quaternion_to_rotation_matrix_numpy(q):
    """Convert quaternion [w,x,y,z] to rotation matrix using numpy."""
    w, x, y, z = q
    
    r00 = 1 - 2*y*y - 2*z*z
    r01 = 2*x*y - 2*w*z
    r02 = 2*x*z + 2*w*y
    
    r10 = 2*x*y + 2*w*z
    r11 = 1 - 2*x*x - 2*z*z
    r12 = 2*y*z - 2*w*x
    
    r20 = 2*x*z - 2*w*y
    r21 = 2*y*z + 2*w*x
    r22 = 1 - 2*x*x - 2*y*y
    
    rotation_matrix = np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ])
    
    return rotation_matrix
