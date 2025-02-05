import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import os
import random
from models.IMUModel2 import IMUTransformer

def load_random_scenes(folder_path, num_scenes=1):
    imu_files = [f for f in os.listdir(folder_path) if "_ms_imu.json" in f]
    scenes = []
    
    selected_files = random.sample(imu_files, num_scenes)
    for imu_file in selected_files:
        scene_id = imu_file.replace("_ms_imu.json", "")
        pose_file = f"{scene_id}_pose.json"
        
        with open(os.path.join(folder_path, imu_file), 'r') as imu_f:
            imu_data = json.load(imu_f)
        with open(os.path.join(folder_path, pose_file), 'r') as pose_f:
            pose_data = json.load(pose_f)
            
        scenes.append((scene_id, imu_data, pose_data))
    
    return scenes

def create_window(data, start_idx, window_size=200):
    end_idx = min(len(data), start_idx + window_size)
    window = data[start_idx:end_idx]
    
    if len(window) < window_size:
        window = window + [window[-1]] * (window_size - len(window))
            
    return window

def get_absolute_pose(relative_poses, initial_pose, window_shift=10):
    current_pos = initial_pose[:3]
    current_quat = initial_pose[3:7]
    trajectory = [initial_pose]
    
    for rel_pose in relative_poses:
        # Get deltas
        pos_delta = rel_pose[:3]
        quat_delta = rel_pose[3:7]
        
        # Transform position delta to world frame
        R = quaternion_to_rotation_matrix(current_quat)
        world_pos_delta = np.matmul(R, pos_delta)
        
        # Update position and orientation
        current_pos = current_pos + world_pos_delta
        current_quat = quaternion_multiply(current_quat, quat_delta)
        current_quat = current_quat / np.linalg.norm(current_quat)
        
        # Store new pose
        new_pose = np.concatenate([current_pos, current_quat])
        trajectory.append(new_pose)
    
    return np.array(trajectory)
def quaternion_slerp(q1, q2, t):
    """Spherical linear interpolation between quaternions."""
    q1 = np.array(q1)
    q2 = np.array(q2)
    
    cos_half_theta = np.dot(q1, q2)
    
    if cos_half_theta < 0:
        q2 = -q2
        cos_half_theta = -cos_half_theta
    
    if cos_half_theta >= 1.0:
        return q1
    
    half_theta = np.arccos(cos_half_theta)
    sin_half_theta = np.sqrt(1.0 - cos_half_theta * cos_half_theta)
    
    if abs(sin_half_theta) < 1e-6:
        return (0.5 * q1) + (0.5 * q2)
    
    ratio_a = np.sin((1 - t) * half_theta) / sin_half_theta
    ratio_b = np.sin(t * half_theta) / sin_half_theta
    
    return (ratio_a * q1) + (ratio_b * q2)

def visualize_trajectories(pred_trajectory, gt_trajectory, scene_id):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    pred_pos = np.array([pose[:3] for pose in pred_trajectory])
    gt_pos = np.array([pose[:3] for pose in gt_trajectory])
    
    ax.plot(pred_pos[:, 0], pred_pos[:, 1], pred_pos[:, 2], 'r-', label='Predicted')
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 'b-', label='Ground Truth')
    
    ax.scatter(pred_pos[0, 0], pred_pos[0, 1], pred_pos[0, 2], c='r', marker='o', label='Start')
    ax.scatter(pred_pos[-1, 0], pred_pos[-1, 1], pred_pos[-1, 2], c='r', marker='s', label='End')
    ax.scatter(gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2], c='b', marker='o')
    ax.scatter(gt_pos[-1, 0], gt_pos[-1, 1], gt_pos[-1, 2], c='b', marker='s')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title(f'Trajectory Comparison - Scene {scene_id}')
    #plt.savefig(f'trajectory_{scene_id}.png')
    plt.show()
    plt.close()

def quaternion_to_rotation_matrix(q):
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
    
    return np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ])

def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    
    return np.array([w, x, y, z])

def run_inference(model, imu_data, pose_data, device, window_size=200, window_shift=10):
    model.eval()
    relative_poses = []
    
    with torch.no_grad():
        for i in range(0, len(pose_data) - window_size, window_shift):
            window = create_window(imu_data, i, window_size)
            
            acc_data = torch.tensor([sample['linear_accel'] for sample in window], 
                                  dtype=torch.float32)[None, ...].to(device)
            gyro_data = torch.tensor([sample['rotation_rate'] for sample in window],
                                   dtype=torch.float32)[None, ...].to(device)
            
            output = model(acc_data, gyro_data)
            relative_pose = output.squeeze().cpu().numpy()
            relative_poses.append(relative_pose)
    
    return relative_poses

def main(imu_data, pose_data, model_path, config, scene_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IMUTransformer(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    window_shift = config.get("window_shift", 10)
    relative_poses = run_inference(model, imu_data, pose_data, device, window_shift=window_shift)
    #ägt_poses = [sample['pos'] + sample['orientation'] for sample in pose_data]
    gt_poses = [sample['pos'] + sample['orientation'] for sample in pose_data[::window_shift]]  
    
    initial_pose = gt_poses[0]
    pred_trajectory = get_absolute_pose(relative_poses, initial_pose, window_shift)
    
    visualize_trajectories(pred_trajectory, gt_poses, scene_id)
    
    return pred_trajectory, gt_poses

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", help="Path to dataset folder")
    parser.add_argument("model_path", help="Path to trained model checkpoint")
    parser.add_argument("config_path", help="Path to config JSON file")
    parser.add_argument("--num_scenes", type=int, default=1, help="Number of random scenes")
    
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as f:
        config = json.load(f)
        
    scenes = load_random_scenes(args.dataset_folder, args.num_scenes)
    
    for scene_id, imu_data, pose_data in scenes:
        print(f"\nProcessing scene: {scene_id}")
        pred_traj, gt_traj = main(imu_data, pose_data, args.model_path, config, scene_id)
        print(f"Trajectory length: {len(pred_traj)} frames")