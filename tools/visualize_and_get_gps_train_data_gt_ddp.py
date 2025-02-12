import argparse
import copy
import os
import pyproj
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import load_checkpoint
# from torchpack import distributed as dist
import torch.distributed as dist
from torchpack.utils.config import configs
#from torchpack.utils.tqdm import tqdm
from tqdm import tqdm
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
import matplotlib.pyplot as plt


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


def main() -> None:
    import os

    os.environ['RANK'] = '0'  # Set this to the rank of the current process
    os.environ['WORLD_SIZE'] = '1'  # Set this to the total number of processes
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'  # Choose any free port

    dist.init_process_group(backend='gloo')  # or 'gloo' for CPU-only

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE")
    parser.add_argument("--mode", type=str, default="gt", choices=["gt", "pred"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--bbox-classes", nargs="+", type=int, default=None)
    parser.add_argument("--bbox-score", type=float, default=None)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--out-dir", type=str, default="viz")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    np.set_printoptions(suppress = True)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=True,
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
        metas = data["metas"].data[0][0]

        token = metas['token']

        sample = nusc.get('sample', token)  # scene_token might actually be a sample_token
        scene_token = sample['scene_token']       # Get the correct scene token
        scene = nusc.get('scene', scene_token)    # Now fetch the scene

        # Get the log associated with the scene
        log = nusc.get('log', scene['log_token'])

        # Retrieve the map name
        map_name = log['location']

        # # Load the map (select the correct map based on your dataset)
        # map_name = "boston-seaport"  # Change based on location (check sample['scene_token'])
        nusc_map = NuScenesMap(dataroot='/data1/data/nuscenes/', map_name=map_name)
        # Get sample from token
        sample = nusc.get('sample', metas['token'])
        # Get sample data for LIDAR_TOP
        sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])

        # Get ego pose
        ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
        # Extract global coordinates
        global_xyz = ego_pose['translation']  # [x, y, z]
        patch_size = 500  # (1km total)


        # Render the map and get figure, axis
        fig, ax = nusc_map.render_map_patch(
            box_coords=[global_xyz[0]-patch_size, global_xyz[1]-patch_size, global_xyz[0]+patch_size, global_xyz[1]+patch_size],  # [x_min, y_min, x_max, y_max]
            figsize=(10, 10),
            layer_names=['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'lane', 'road_segment', 'road_block'],
            render_egoposes_range = False,
            render_legend = False,
            alpha=0.5
        )

        name = "{}-{}".format(metas["timestamp"], metas["token"])


        save_path = 'all_'+args.split+'_basemaps/'+name+"_base_map_image.png"  # Change this to your desired save location
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Close the figure to free memory
        metas_save_path = 'all_'+args.split+'_metas/'+name+"_metas.npy"  # Change this to your desired save location
        np.save(metas_save_path, metas)


        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
        elif args.mode == "pred" and "masks_bev" in outputs[0]:
            masks = outputs[0]["masks_bev"].numpy()
            masks = masks >= args.map_score
        else:
            masks = None

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}_generated_map_image.png"),
                masks,
                classes=cfg.map_classes,
            )


if __name__ == "__main__":
    main()
