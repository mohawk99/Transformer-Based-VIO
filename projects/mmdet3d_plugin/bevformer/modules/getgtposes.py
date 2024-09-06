import json

def getgtposes(sample_token):


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

def main():
    # Example usage
    sample_token = 'ba2065b0769c4361b071ef028605d7bb'

    try:
        pose = getgtposes(sample_token)
        print(f"Pose: {pose}")
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
