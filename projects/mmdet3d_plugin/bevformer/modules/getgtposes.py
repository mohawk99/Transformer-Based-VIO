import json
from nuscenes.nuscenes import NuScenes
import torch


def getimuposes(sample_token):
    sample_json_path = "/home/mohak/Thesis/BEVFormer/data/nuscenes/v1.0-trainval/sample.json"
    ego_pose_json_path = "/home/mohak/Thesis/BEVFormer/data/nuscenes/v1.0-trainval/ego_pose.json"
    # Load sample.json
    with open(sample_json_path) as f:
        sample_data = json.load(f)

    # Load ego_pose.json
    with open(ego_pose_json_path) as f:
        ego_pose_data = json.load(f)

    # Find the timestamp for the specific sample token
    sample_timestamp = None
    for sample in sample_data:
        if sample['token'] == sample_token:
            sample_timestamp = sample['timestamp']
            break

    if sample_timestamp is None:
        raise ValueError("Sample token not found!")

    # Find the closest matching timestamp in ego_pose.json
    closest_ego_pose = None
    for ego_pose in ego_pose_data:
        if ego_pose['timestamp'] == sample_timestamp:
            closest_ego_pose = ego_pose
            break

    if closest_ego_pose is None:
        raise ValueError("Matching timestamp in ego_pose.json not found!")

    # Extract rotation and translation
    rotation = closest_ego_pose['rotation']
    translation = closest_ego_pose['translation']

    pose = translation[:3] + rotation

    return pose

def getvoposes(nusc, img_metas, sensors=None):
    if sensors is None:
        sensors = ['CAM_FRONT']
    stacked_poses = []

    for i, img_meta_dict in enumerate(img_metas):
        for key, img_meta in img_meta_dict.items():

            # Check if 'sample_idx' exists
            if 'sample_idx' not in img_meta:
                print(f"Error: 'sample_idx' not found in img_meta[{key}]")
                continue

            # Extract sample token
            sample_token = img_meta['sample_idx']

            # Get sample
            sample = nusc.get('sample', sample_token)

            for sensor in sensors:
                if sensor not in sample['data']:
                    print(f"Warning: Sensor {sensor} not found in sample data.")
                    continue

                # Get camera data and ego pose
                cam_token = sample['data'][sensor]
                cam_data = nusc.get('sample_data', cam_token)
                ego_pose_token = cam_data['ego_pose_token']
                ego_pose = nusc.get('ego_pose', ego_pose_token)

                translation = ego_pose['translation']  # [x, y, z]
                rotation = ego_pose['rotation']        # [w, x, y, z]
                stacked_poses.append(translation + rotation)  # Shape: [7]

    # Convert to a single tensor
    stacked_tensor = torch.tensor(stacked_poses, dtype=torch.float32)  # Shape: [B, 7]
    return stacked_tensor

# def main():
#     # Example usage
#     sample_token = 'ba2065b0769c4361b071ef028605d7bb'

#     try:
#         pose = getgtposes(sample_token)
#         print(f"Pose: {pose}")
#     except ValueError as e:
#         print(e)

# if __name__ == "__main__":
#     main()
