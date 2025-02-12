from tqdm import tqdm
import json
import cv2
import numpy as np
from pathlib import Path
import time
import multiprocessing as mp

def extract_timestamp(filename):
    return filename.split('-')[0]

def get_neighbors(filenames):
    sorted_files = sorted(filenames, key=extract_timestamp)
    total_files = len(sorted_files)
    neighbors_dict = {}

    for i, fname in enumerate(sorted_files):
        if total_files < 10:
            neighbors_dict[fname] = {
                "index": i,
                "neighbors": sorted_files
            }
        else:
            start = max(0, i - 5)
            end = min(total_files, i + 6)
            while end - start < 10:
                if start > 0:
                    start -= 1
                elif end < total_files:
                    end += 1
                else:
                    break
            neighbors_dict[fname] = {
                "index": i,
                "neighbors": sorted_files[start:end]
            }
    
    return neighbors_dict

def stitch_bev_maps(image_paths, output_path="stitched_output.png"):
    images = [cv2.imread(str(p)) for p in image_paths]
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    stitcher.setRegistrationResol(1)
    stitcher.setCompositingResol(1)
    stitcher.setPanoConfidenceThresh(0.6)

    status, stitched = stitcher.stitch(images)

    if status == cv2.Stitcher_OK:
        stitched = cv2.resize(stitched, (2048, 1024))
        cv2.imwrite(output_path, stitched)
        return stitched
    else:
        print(f"Stitching failed with status {status}")
        return None

def process_scene(scene_data):
    scene, file_list = scene_data
    folder_name = "/data1/all_val_maps_gt/map/"
    output_stitched_maps_folder = "/data1/all_val_maps_gt_stitched/"

    try:
        file_list = [file.split("_metas.npy")[0] for file in file_list]
        neighbors = get_neighbors(file_list)

        for fname, neighbor_list in neighbors.items():
            neighbor_list_files = [folder_name + item + "_generated_map_image.png" for item in neighbor_list["neighbors"]]
            stitched_map = stitch_bev_maps(
                neighbor_list_files,
                output_path=output_stitched_maps_folder + fname + "_stitched_map.png"
            )
    except Exception as e:
        print(f"Error processing scene {scene}: {e}")

if __name__ == "__main__":
    with open("scene_to_file_map_val.json", "r") as f:
        scene_to_file_map = json.load(f)

    num_processes = mp.cpu_count()-4  # Use all available CPU cores
    pool = mp.Pool(processes=num_processes)

    scene_data = list(scene_to_file_map.items())
    
    for _ in tqdm(pool.imap_unordered(process_scene, scene_data), total=len(scene_data), desc="Processing scenes"):
        pass

    pool.close()
    pool.join()
