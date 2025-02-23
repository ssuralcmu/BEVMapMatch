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
import numpy as np
import io
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg
import time

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

def render_map_patch_to_array(nusc_map, *args, dpi=200, **kwargs):
    """
    Wrapper function to convert render_map_patch output to numpy array.
    
    :param nusc_map: NuScenesMap object
    :param args: Positional arguments to pass to render_map_patch
    :param kwargs: Keyword arguments to pass to render_map_patch
    :return: Numpy array of the rendered map patch
    """
    # Call the original render_map_patch method
    fig, ax = nusc_map.render_map_patch(*args, **kwargs)

    bbox_coords = kwargs.get('box_coords')
    ax.set_xlim(bbox_coords[0], bbox_coords[2])
    ax.set_ylim(bbox_coords[1], bbox_coords[3])

    # Remove axis ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.axis('off')

    fig.set_dpi(dpi)
    
    # Save the figure to a buffer
    buf = io.BytesIO()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    
    # Close the figure to free up memory
    fig.clf()
    
    # Convert buffer to numpy array
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)

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


            # Define the map patch around the ego vehicle
            patch_size = 500  # 200 meters in each direction (1km total)
            
            # Perturb the global coordinates
            perturbation = np.random.randint(-1*(patch_size-100), (patch_size-100), size=2)
            global_xyz[0] += perturbation[0]
            global_xyz[1] += perturbation[1]

            metas['perturbation'] = perturbation
            metas['new_global_xyz'] = global_xyz

            dpi = 200  # Set an appropriate DPI value
            #figsize = (2*2*patch_size/dpi, 2*2*patch_size/dpi)  # Calculate figsize in inches
            figsize = (2*2*patch_size/dpi, 2*2*patch_size/dpi)  # Calculate figsize in inches

            # Render the map and get figure, axis
            map_array = render_map_patch_to_array(nusc_map,
                box_coords=[global_xyz[0]-patch_size, global_xyz[1]-patch_size, global_xyz[0]+patch_size, global_xyz[1]+patch_size],  # [x_min, y_min, x_max, y_max]
                figsize=figsize,
                layer_names=['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'lane', 'road_segment', 'road_block'],
                render_egoposes_range = False,
                render_legend = False,
                alpha=0.5,
                dpi=dpi
            )
            # print("Saving map figure...")

            # Save the figure

            name = "{}-{}".format(metas["timestamp"], metas["token"])


            save_path = 'all_'+args.split+'_basemaps_v2/'+name+"_base_map_image.png"  # Change this to your desired save location
            image = Image.fromarray(map_array)

            # Save the image to a file
            image.save(save_path)
            # print(f"Map image saved at: {save_path}")
            
            #Save metas variable to file
            # print("Saving metas variable...")
            metas_save_path = 'all_'+args.split+'_metas_v2/'+name+"_metas.npy"  # Change this to your desired save location
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
