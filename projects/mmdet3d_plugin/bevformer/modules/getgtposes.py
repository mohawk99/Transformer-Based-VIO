import json
from nuscenes.nuscenes import NuScenes

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

def getvoposes(nusc, img_meta, sensor=None):

    if sensor is None:
        sensor = 'CAM_FRONT'

    sample_token = img_meta['sample_idx']
    
    sample = nusc.get('sample', sample_token)
    cam_token = sample['data'][sensor]
    cam_data = nusc.get('sample_data', cam_token)
    
    ego_pose_token = cam_data['ego_pose_token']
    ego_pose = nusc.get('ego_pose', ego_pose_token)
    
    translation = ego_pose['translation']  # [x, y, z]
    rotation = ego_pose['rotation']        # Quaternion [w, x, y, z]
    
    return translation, rotation

def print_ego_pose_for_all_sensors(nusc, img_meta):
    sample_token = img_meta['sample_idx']
    sample = nusc.get('sample', sample_token)

    print(f"Ego pose for sample token: {sample_token}")
    for sensor, cam_token in sample['data'].items():
        cam_data = nusc.get('sample_data', cam_token)
        ego_pose_token = cam_data['ego_pose_token']
        ego_pose = nusc.get('ego_pose', ego_pose_token)
        
        translation = ego_pose['translation']  # [x, y, z]
        rotation = ego_pose['rotation']        # Quaternion [w, x, y, z]
        print(f"Sensor: {sensor}")
        print(f"  Translation: {translation}")
        print(f"  Rotation: {rotation}")


# def main():
#     # Initialize NuScenes
#     nusc = NuScenes(version='v1.0-trainval', dataroot='/home/mohak/Thesis/PanoOcc/data/nuscenes', verbose=True)
    
#     # Example `img_metas_list` (replace with your actual metadata)
#     img_metas_list = [
#         {
#             'filename': [
#                 './data/nuscenes/samples/CAM_FRONT/n015-2018-11-14-19-09-14+0800__CAM_FRONT__1542194035112460.jpg',
#                 './data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-11-14-19-09-14+0800__CAM_FRONT_RIGHT__1542194035120339.jpg',
#                 './data/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-11-14-19-09-14+0800__CAM_FRONT_LEFT__1542194035104844.jpg',
#                 './data/nuscenes/samples/CAM_BACK/n015-2018-11-14-19-09-14+0800__CAM_BACK__1542194035137525.jpg',
#                 './data/nuscenes/samples/CAM_BACK_LEFT/n015-2018-11-14-19-09-14+0800__CAM_BACK_LEFT__1542194035147423.jpg',
#                 './data/nuscenes/samples/CAM_BACK_RIGHT/n015-2018-11-14-19-09-14+0800__CAM_BACK_RIGHT__1542194035127893.jpg'
#             ],
#             'ori_shape': [(450, 800, 3)] * 6,
#             'img_shape': [(480, 800, 3)] * 6,
#             'lidar2img': None,
#             'lidar2cam': None,
#             'pad_shape': [(480, 800, 3)] * 6,
#             'scale_factor': 1.0,
#             'box_mode_3d': None,
#             'box_type_3d': None,
#             'img_norm_cfg': None,
#             'sample_idx': 'ba2065b0769c4361b071ef028605d7bb',
#             'scene_token': 'f0f7132494bc4045a21868aca13b56f9'
#         }
#     ]
    
#     # Print ego pose for all sensors in the first metadata entry
#     print_ego_pose_for_all_sensors(nusc, img_metas_list[0])

# if __name__ == "__main__":
#     main()

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
