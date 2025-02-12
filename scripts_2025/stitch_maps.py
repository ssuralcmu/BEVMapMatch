from tqdm import tqdm
import json
import cv2
import numpy as np
from pathlib import Path
import time

with open("scene_to_file_map_val.json", "r") as f:
    scene_to_file_map = json.load(f)


def extract_timestamp(filename):
    return filename.split('-')[0]

def get_neighbors(filenames):
    # Extract timestamps and sort filenames based on them
    sorted_files = sorted(filenames, key=extract_timestamp)
    total_files = len(sorted_files)

    file_map = {fname: idx for idx, fname in enumerate(sorted_files)}
    neighbors_dict = {}

    for i, fname in enumerate(sorted_files):
        if total_files < 10:
            neighbors_dict[fname] = {
                "index": i,
                "neighbors": sorted_files  # Return all files if total is less than 10
            }
        else:
            start = max(0, i - 5)
            end = min(total_files, i + 6)  # Initial window

            # Ensure at least 10 files
            while end - start < 10:
                if start > 0:
                    start -= 1
                elif end < total_files:
                    end += 1
                else:
                    break  # No more adjustments possible
            
            neighbors_dict[fname] = {
                "index": i,
                "neighbors": sorted_files[start:end]
            }
    
    return neighbors_dict



def stitch_bev_maps(image_paths, output_path="stitched_output.png"):
    # Modified sorting key to handle numeric prefixes    
    images = [cv2.imread(str(p)) for p in image_paths]

    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    stitcher.setRegistrationResol(1) #1 for GT, 1 for pred
    stitcher.setCompositingResol(1) #1 for GT, 1 for pred
    stitcher.setPanoConfidenceThresh(0.6) #0.6 for GT, 0.7 for pred

    # start_time = time.time()    
    status, stitched = stitcher.stitch(images)
    # end_time=time.time()

    # print(f"Stitching took {end_time - start_time} seconds")

    # print(stitched)
    # print(stitched.shape)

    #Resize the stitched image to 1024x2048
    stitched = cv2.resize(stitched, (2048, 1024))

    # print(stitched)
    # print(np.unique(stitched))

    if status == cv2.Stitcher_OK:
        cv2.imwrite(output_path, stitched)
        return stitched
    else:
        print(f"Stitching failed with status {status}")
        return None

folder_name="/data1/all_val_maps_gt/map/"

output_stitched_maps_folder = "/data1/all_val_maps_gt_stitched/"

for scene, file_list in tqdm(scene_to_file_map.items(), desc="Processing scenes"):
    # print("Length of file_list:", len(file_list))

    try:

        #Iterate through file_list and extract the name of the file - the part before "_metas.npy"
        file_list = [file.split("_metas.npy")[0] for file in file_list]

        #all_val_maps_gt/map/ folder has a file called file_list[i]_generated_map_image.png for each file in file_list

        #For every such file, get 10 maps or fewer if 10 are not there around it. Towards the ends, there might be fewer than 10 maps, so get extra from the middle to make the count 10.
        neighbors = get_neighbors(file_list)

        
        for fname, neighbor_list in neighbors.items():
            neighbor_list_files = [folder_name+item + "_generated_map_image.png" for item in neighbor_list["neighbors"]]
            # print(f"File: {fname}")
            # print(f"Neighbors: {neighbor_list_files}\n")
            # Usage

            stitched_map = stitch_bev_maps(
                neighbor_list_files,
                output_path=output_stitched_maps_folder+fname+"_stitched_map.png"
            )

    except Exception as e:
        print("Scene:", scene)  
        print(f"Error processing scene {scene}: {e}")
    except KeyboardInterrupt:
        print("Process interrupted by user.")
        break