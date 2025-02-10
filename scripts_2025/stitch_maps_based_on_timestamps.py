import os
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap


nusc = NuScenes(version='v1.0-trainval', dataroot='/data1/data/nuscenes/', verbose=True)

# Define the folder path
folder_path = '../all_val_metas'

# Get a list of all .npy files in the folder
npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]
npy_files.sort()

print("loaded")

# Iterate through the first 100 .npy files
for i, file in enumerate(npy_files[:100]):
    file_path = os.path.join(folder_path, file)
    data = np.load(file_path, allow_pickle=True).item()
    if i < 10:
        print((data['timestamp']))
        print((data['token']))
        sample = nusc.get('sample', data['token'])  # scene_token might actually be a sample_token
        scene_token = sample['scene_token']       # Get the correct scene token
        scene = nusc.get('scene', scene_token)    # Now fetch the scene

        # Get the log associated with the scene
        log = nusc.get('log', scene['log_token'])

        # Retrieve the map name
        map_name = log['location']

        print("Map Name:", map_name)

        # # metas
        # print("Debuggging inside visuliase.py ******* ")

        # # Load the map (select the correct map based on your dataset)
        # map_name = "boston-seaport"  # Change based on location (check sample['scene_token'])
        nusc_map = NuScenesMap(dataroot='/data1/data/nuscenes/', map_name=map_name)

        # Get sample from token
        sample = nusc.get('sample', data['token'])

        # Get sample data for LIDAR_TOP
        sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        # Get ego pose
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])

        # print("Ego pose:", ego_pose)

        # Extract global coordinates
        global_xyz = ego_pose['translation']  # [x, y, z]
        print("Global coordinates:", global_xyz)

    else:
        break
        # print(key for _,key in data.items())

