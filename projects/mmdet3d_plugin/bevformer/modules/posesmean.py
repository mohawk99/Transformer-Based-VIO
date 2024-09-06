import numpy as np
from nuscenes.nuscenes import NuScenes

# Load NuScenes dataset
nusc = NuScenes(version='v1.0-trainval', dataroot='/home/mohak/Thesis/BEVFormer/data/nuscenes', verbose=True)

# List to store all x and y translation values
all_x = []
all_y = []
scene = nusc.scene[0]
# Iterate over every scene
#for scene in nusc.scene:
first_sample_token = scene['first_sample_token']
current_sample_token = first_sample_token
i=1
while current_sample_token != '':
    # Get the current sample
    my_sample = nusc.get('sample', current_sample_token)
    
    # Extract the data for CAM_FRONT
    sensor = 'CAM_FRONT'
    cam_front_data = nusc.get('sample_data', my_sample['data'][sensor])
    
    # Get the ego pose using the ego_pose_token
    ego_pose = nusc.get('ego_pose', cam_front_data['ego_pose_token'])
    
    # Extract translation values (x and y)
    x, y, _ = ego_pose['translation']
    
    # Append to the lists
    all_x.append(x)
    all_y.append(y)
    
    # Move to the next sample
    current_sample_token = my_sample['next']
    i+=1

# Convert lists to numpy arrays for easier calculation
all_x = np.array(all_x)
all_y = np.array(all_y)

print("Original x translation values:", all_x)
print("Original y translation values:", all_y)


# Calculate mean and standard deviation
mean_x = np.mean(all_x)
std_x = np.std(all_x)
mean_y = np.mean(all_y)
std_y = np.std(all_y)

print(f"Mean of x: {mean_x}, Standard Deviation of x: {std_x}")
print(f"Mean of y: {mean_y}, Standard Deviation of y: {std_y}")
print(f"total samples : {i}")

if std_x != 0:
    # Z-normalize the x values
    z_normalized_x = [(x - mean_x) / std_x for x in all_x]
else:
    z_normalized_x = ["Undefined (std_x=0)" for x in all_x]

if std_y != 0:
    # Z-normalize the y values
    z_normalized_y = [(y - mean_y) / std_y for y in all_y]
else:
    z_normalized_y = ["Undefined (std_y=0)" for y in all_y]

# Print Z-normalized translation values
print("Z-normalized x translation values:", z_normalized_x)
print("Z-normalized y translation values:", z_normalized_y)
