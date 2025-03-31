from tqdm import tqdm
import json
import cv2
import numpy as np
from pathlib import Path
import time
import multiprocessing as mp
import glob
from stitching.images import Images
from stitching import Stitcher
import matplotlib.pyplot as plt

def extract_timestamp(filename):
    return filename.split('-')[0]

def get_neighbors(filenames):
    sorted_files = sorted(filenames, key=extract_timestamp)
    total_files = len(sorted_files)
    neighbors_dict = {}

    for i, fname in enumerate(sorted_files):
        if total_files < 20:
            neighbors_dict[fname] = {
                "index": i,
                "neighbors": sorted_files
            }
        else:
            start = max(0, i - 10)
            end = min(total_files, i + 10)
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
    
    #Print length of neighbors_dict
    # print("Length of neighbors_dict: ", len(neighbors_dict))
    return neighbors_dict

def apply_homography_to_point(H, point):
    x, y = point
    p = np.array([x, y, 1])
    p_transformed = H @ p
    p_transformed /= p_transformed[2]  # Normalize to make the third coordinate 1
    return p_transformed[0], p_transformed[1]

def stitch_bev_maps_and_get_localization(metas_path, image_paths, output_path="stitched_output.png"):
    metas = np.load(metas_path, allow_pickle=True).item()
    # Load the original images using OpenCV so we can compute their centers.
    original_images = [cv2.imread(path) for path in image_paths]

    # Initialize the Stitcher with affine parameters 
    stitcher = Stitcher(
        confidence_threshold=0.6,
        medium_megapix=1,
        final_megapix=1,
        detector='orb',
        matcher_type='affine',
        estimator='affine',
        adjuster='affine',
        warper_type='affine',
        compensator='no',
        wave_correct_kind='no'
    )

    # Initialize the result with the first image
    result = cv2.imread(image_paths[0])

    #Summary is that use num_per_stitch=2 and 50. Or maybe 20? Take the better of the two. If one errors out (smaller one), take bigger. Else, Take smaller. 

    try:   
        num_per_stitch=50#4 #2#50
        # Stitch images in groups of n
        for i in range(1, len(image_paths), num_per_stitch):
            next_images = [result] + [cv2.imread(path) for path in image_paths[i:i+num_per_stitch]]
            images_obj = Images.of(next_images)
            final_imgs = list(images_obj.resize(Images.Resolution.FINAL))
            stitched = stitcher.stitch(final_imgs)
            if stitched is None:
                # print(f"Failed to stitch images {i}-{min(i+num_per_stitch, len(image_paths))}. Using previous result.")
                continue
            
            # Update the result
            result = stitched
        # print("Worked with 2")
    except Exception as e:
        try:
            num_per_stitch=2
            for i in range(1, len(image_paths), num_per_stitch):
                next_images = [result] + [cv2.imread(path) for path in image_paths[i:i+num_per_stitch]]
                images_obj = Images.of(next_images)
                final_imgs = list(images_obj.resize(Images.Resolution.FINAL))
                stitched = stitcher.stitch(final_imgs)
                if stitched is None:
                    # print(f"Failed to stitch images {i}-{min(i+num_per_stitch, len(image_paths))}. Using previous result.")
                    continue
                
                # Update the result
                result = stitched

            # print("Worked with 100")
        except Exception as e:
            # print(f"Failed to stitch images {i}-{min(i+num_per_stitch, len(image_paths))}. Using previous result.")
            return
    
    # Save and display the final result
    cv2.imwrite(output_path, result)

    individual_image=original_images[len(original_images)-1]
    stitched_image=result

    # Convert to grayscale
    stitched_image_gray = cv2.cvtColor(stitched_image, cv2.COLOR_BGR2GRAY)
    individual_image_gray = cv2.cvtColor(individual_image, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create(nfeatures=2000)
    keypoints1, descriptors1 = orb.detectAndCompute(stitched_image_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(individual_image_gray, None)

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) >= 4:
        src_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Compute homography
        matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        # Transform corners of second image to find its location in the first image
        h, w = individual_image_gray.shape
        corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, matrix)

        # Calculate center of transformed corners
        center_x = np.mean(transformed_corners[:, 0, 0])
        center_y = np.mean(transformed_corners[:, 0, 1])
        # print(f"Center of second image in first image: ({center_x}, {center_y})")

        metas["stitched_image_center"] = [center_x,center_y]
        np.save(metas_path, metas)
    else:
        # print("Not enough matches to determine location.")
        metas["stitched_image_center"] = None
        np.save(metas_path, metas)

# def stitch_bev_maps(image_paths, output_path="stitched_output.png"):
#     images = [cv2.imread(str(p)) for p in image_paths]
#     stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
#     stitcher.setRegistrationResol(1)
#     stitcher.setCompositingResol(1)
#     stitcher.setPanoConfidenceThresh(0.6)

#     status, stitched = stitcher.stitch(images)

#     if status == cv2.Stitcher_OK:
#         # stitched = cv2.resize(stitched, (2048, 1024))
#         cv2.imwrite(output_path, stitched)
#         return stitched
#     else:
#         print(f"Stitching failed with status {status}")
#         return None

def process_scene(scene_data):
    try:
        scene, file_list = scene_data
        folder_name = "/data1/all_"+split+"_maps_gt_v2/map/"
        output_stitched_maps_folder = "/data1/all_"+split+"_maps_gt_v2_stitched/"

        # Create output folder if it doesn't exist
        Path(output_stitched_maps_folder).mkdir(parents=True, exist_ok=True)

        # try:
        file_list = [file.split("_metas.npy")[0] for file in file_list]
        neighbors = get_neighbors(file_list)

        for fname, neighbor_list in neighbors.items():
            neighbor_list_files = [folder_name + item + "_generated_map_image.png" for item in neighbor_list["neighbors"]]
            stitched_map = stitch_bev_maps_and_get_localization(
                "../all_"+split+"_metas_v2/"+fname+"_metas.npy",
                neighbor_list_files,
                output_path=output_stitched_maps_folder + fname + "_stitched_map.png"
            )
    except Exception as e:
        print(f"Error processing scene {scene}: {e}")
    # except Exception as e:
    #     print(f"Error processing scene {scene}: {e}")

split = "val"

if __name__ == "__main__":
    with open("scene_to_file_map_v2_"+split+".json", "r") as f:
        scene_to_file_map = json.load(f)

    num_processes = mp.cpu_count()-8  # Use all available CPU cores
    pool = mp.Pool(processes=num_processes)

    scene_data = list(scene_to_file_map.items())
    
    for _ in tqdm(pool.imap_unordered(process_scene, scene_data), total=len(scene_data), desc="Processing scenes"):
        pass

    pool.close()
    pool.join()

