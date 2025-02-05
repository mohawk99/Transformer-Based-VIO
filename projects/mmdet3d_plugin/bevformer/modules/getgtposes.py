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


def getvoposes(nusc, img_metas, json_path):

    stacked_poses = []

    for i, img_meta_dict in enumerate(img_metas):
        for key, img_meta in img_meta_dict.items():
            # Ensure 'sample_idx' exists
            if 'sample_idx' not in img_meta:
                print(f"Error: 'sample_idx' not found in img_meta[{key}]")
                continue

            # Extract sample token
            sample_token = img_meta['sample_idx']

            # Extract scene token from the sample token
            sample = nusc.get('sample', sample_token)
            scene_token = sample['scene_token']


                # Get normalized translation and original orientation
            try:
                pose_data = get_normalized_pose(json_path, scene_token, sample_token)
            except ValueError as e:
                print(f"Error retrieving pose data: {e}")
                continue

            # Append the pose data to the list
            stacked_poses.append(pose_data)

    # Convert to a single tensor
    stacked_tensor = torch.tensor(stacked_poses, dtype=torch.float32)  # Shape: [B, 7]
    return stacked_tensor





def normalize_ego_poses(nusc, output_path="norm_gt_poses.json"):

    # Dictionary to store normalized data for all scenes
    all_normalized_data = {}

    # Iterate through all scenes
    for scene in nusc.scene:
        scene_token = scene["token"]  # Use scene token instead of scene name
        first_sample_token = scene["first_sample_token"]

        # Store all ego poses for the current scene
        ego_poses = []
        sample_tokens = []
        orientations = []  # To store orientations (rotation)
        current_sample_token = first_sample_token

        # Iterate through all samples in the scene
        while current_sample_token:
            sample = nusc.get("sample", current_sample_token)
            ego_pose = nusc.get("ego_pose", sample["data"]["CAM_FRONT"])
            
            # Store translation, orientation, and sample token
            ego_poses.append(ego_pose["translation"])
            orientations.append(ego_pose["rotation"])
            sample_tokens.append(current_sample_token)
            
            # Move to the next sample
            current_sample_token = sample["next"] if sample["next"] else None

        # Calculate min and max translation values for normalization
        translations = list(zip(*ego_poses))  # Transpose for x, y, z
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
            for pose in ego_poses
        ]

        # Store results for the scene
        all_normalized_data[scene_token] = {
            "normalization_values": {
                "min_translation": min_translation,
                "max_translation": max_translation,
            },
            "samples": {
                sample_token: {
                    "normalized_translation": normalized_translation,
                    "original_orientation": orientation,
                }
                for sample_token, normalized_translation, orientation in zip(
                    sample_tokens, normalized_translations, orientations
                )
            },
        }

    # Save the normalized data to a JSON file
    with open(output_path, "w") as f:
        json.dump(all_normalized_data, f, indent=4)

    print(f"Normalized ego poses saved to {output_path}")
    #return all_normalized_data

def get_normalized_pose(json_path, scene_token, sample_token):

    # Load the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Retrieve scene data using the scene token
    scene_data = data.get(scene_token, {})
    if not scene_data:
        raise ValueError(f"Scene token '{scene_token}' not found in the JSON data.")

    # Retrieve sample data
    sample_data = scene_data.get("samples", {}).get(sample_token, {})
    if not sample_data:
        raise ValueError(f"Sample token '{sample_token}' not found in scene with token '{scene_token}'.")

    # Extract normalized translation and original orientation
    normalized_translation = sample_data["normalized_translation"]
    original_orientation = sample_data["original_orientation"]
    
    # Return combined data
    return normalized_translation + original_orientation

# def main():
#     """
#     Main function to normalize poses for all scenes and test retrieval for a specific sample.
#     """
#     # Initialize the NuScenes dataset
#     nusc = NuScenes(version='v1.0-trainval', dataroot='/home/mohak/Thesis/PanoOcc/data/occ3d-nus/', verbose=True)

#     # Normalize ego poses for all scenes and save to file
#     print("Normalizing ego poses for all scenes...")
#     normalize_ego_poses(nusc)

#     # Test retrieval for a specific scene token and sample token
#     test_scene_token = "d25718445d89453381c659b9c8734939"  # Replace with a valid scene token
#     test_sample_token = "29796060110c4163b07f06eff4af0753"  # Replace with a valid sample token
#     try:
#         pose_data = get_normalized_pose("norm_gt_poses.json", test_scene_token, test_sample_token)
#         print(f"Normalized pose data for scene '{test_scene_token}', sample '{test_sample_token}': {pose_data}")
#     except ValueError as e:
#         print(e)

# # Run the main function
# if __name__ == "__main__":
#     main()
