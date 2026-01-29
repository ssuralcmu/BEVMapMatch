import argparse
import copy
import os
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from torchpack import distributed as dist
from torchpack.utils.config import configs
#from torchpack.utils.tqdm import tqdm
from tqdm import tqdm
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
import matplotlib.pyplot as plt
import time
from pyquaternion import Quaternion

def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for k, val in enumerate(obj):
            obj[k] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj

def build_segmentation_basemap(nusc_map, center_xy, patch_size, classes):
    mappings = {}
    for name in classes:
        if name == "drivable_area*":
            mappings[name] = ["road_segment", "lane"]
        elif name == "divider":
            mappings[name] = ["road_divider", "lane_divider"]
        else:
            mappings[name] = [name]

    layer_names = []
    for name in mappings:
        layer_names.extend(mappings[name])
    layer_names = list(set(layer_names))

    canvas_size = (patch_size * 4, patch_size * 4)
    patch_box = (center_xy[0], center_xy[1], patch_size * 2, patch_size * 2)
    masks = nusc_map.get_map_mask(
        patch_box=patch_box,
        patch_angle=0.0,
        layer_names=layer_names,
        canvas_size=canvas_size,
    )
    masks = masks.transpose(0, 2, 1)
    masks = masks.astype(np.bool)

    labels = np.zeros((len(classes), *canvas_size), dtype=np.bool)
    for k, name in enumerate(classes):
        for layer_name in mappings[name]:
            index = layer_names.index(layer_name)
            labels[k, masks[index]] = 1
    return labels

def main() -> None:
    dist.init()

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    parser.add_argument("--start-idx", type=int, default=None)
    parser.add_argument("--end-idx", type=int, default=None)
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    np.set_printoptions(suppress = True)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    
    from torch.utils.data import Subset
    start=args.start_idx
    end=args.end_idx
    subset_indices = list(range(start, end))
    subset_dataset = Subset(dataset, subset_indices)
    dataflow = build_dataloader(
        subset_dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # build the model and load checkpoint
    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")

        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
        )
        model.eval()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.map_expansion.map_api import NuScenesMap

    # Load NuScenes dataset
    nusc = NuScenes(version='v1.0-trainval', dataroot='/data1/data/nuscenes/', verbose=True)

    print("Length of dataset: ", len(dataset))
    for data in tqdm(dataflow):
        try:
            # print("Data keys:",data.keys())
            # print("Data metas:",data["metas"].data[0][0].keys())
            metas = data["metas"].data[0][0]

            token = metas['token']

            sample = nusc.get('sample', token)  # scene_token might actually be a sample_token
            scene_token = sample['scene_token']       # Get the correct scene token
            scene = nusc.get('scene', scene_token)    # Now fetch the scene

            # Get the log associated with the scene
            log = nusc.get('log', scene['log_token'])

            # Retrieve the map name
            map_name = log['location']

            # print("Map Name:", map_name)

            # # metas
            # print("Debuggging inside visuliase.py ******* ")

            # # Load the map (select the correct map based on your dataset)
            # map_name = "boston-seaport"  # Change based on location (check sample['scene_token'])
            nusc_map = NuScenesMap(dataroot='/data1/data/nuscenes/', map_name=map_name)

            # Get sample from token
            sample = nusc.get('sample', metas['token'])

            # Get sample data for LIDAR_TOP
            sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

            # Get ego pose
            ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])

            # print("Ego pose:", ego_pose)

            # Extract global coordinates
            global_xyz = ego_pose['translation']  # [x, y, z]

            ego_yaw = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0]

            # Define the map patch around the ego vehicle
            patch_size = 250  # 500 m x 500 m map
            
            # Perturb the global coordinates
            perturbation = np.random.randint(-1*(patch_size-50), (patch_size-50), size=2) # Variability increased to 200 m x 200 m. This makes things more complex.

            global_xyz[0] += perturbation[0]
            global_xyz[1] += perturbation[1]

            metas['perturbation'] = perturbation
            metas['new_global_xyz'] = global_xyz
            metas['map_patch_angle'] = 0.0
            metas['map_relative_yaw'] = ego_yaw
            
            map_masks = build_segmentation_basemap(
                nusc_map,
                (global_xyz[0], global_xyz[1]),
                patch_size,
                cfg.map_classes,
            )

            # Save the figure

            name = "{}-{}".format(metas["timestamp"], metas["token"])


            save_path = 'all_train_basemaps_segmented_v3/'+name+"_base_map_image.png"  # Change this to your desired save location
            metas_save_path = 'all_train_metas_v3/'+name+"_metas.npy"  # Change this to your desired save
            #Make directory if it does not exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            #Check if file already exists
            if os.path.exists(save_path) and os.path.exists(metas_save_path):
                continue

            visualize_map(
                save_path,
                map_masks,
                classes=cfg.map_classes,
            )

            plt.close('all')
            
            #Save metas variable to file
            # print("Saving metas variable...")
            #Make directory if it does not exist
            os.makedirs(os.path.dirname(metas_save_path), exist_ok=True)
            np.save(metas_save_path, metas)


            if args.mode == "pred":
                with torch.inference_mode():
                    outputs = model(**data)


            if args.mode == "gt" and "gt_masks_bev" in data:
                masks = data["gt_masks_bev"].data[0].numpy()
                masks = masks.astype(np.bool)
            elif args.mode == "pred" and "masks_bev" in outputs[0]:
                masks = outputs[0]["masks_bev"].numpy()
                masks = masks >= args.map_score
            else:
                masks = None

            if masks is not None:
                # print("Saving generated map")
                # print(os.path.join(args.out_dir, "map", f"{name}.png"))
                visualize_map(
                    os.path.join(args.out_dir, "map", f"{name}_generated_map_image.png"),
                    masks,
                    classes=cfg.map_classes,
                )
        except MemoryError:
            print("Memory Error")
            time.sleep(100)
            continue


if __name__ == "__main__":
    main()
