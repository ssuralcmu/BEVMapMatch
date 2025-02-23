import argparse
import copy
import os
import pyproj
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from tqdm import tqdm
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from torchpack.utils.config import configs
import multiprocessing as mp

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

def process_sample(data, args, cfg, nusc):
    metas = data["metas"].data[0][0]
    token = metas['token']
    sample = nusc.get('sample', token)
    scene_token = sample['scene_token']
    scene = nusc.get('scene', scene_token)
    log = nusc.get('log', scene['log_token'])
    map_name = log['location']
    nusc_map = NuScenesMap(dataroot='/data1/data/nuscenes/', map_name=map_name)

    sample = nusc.get('sample', metas['token'])
    sample_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
    global_xyz = ego_pose['translation']
    patch_size = 500

    fig, ax = nusc_map.render_map_patch(
        box_coords=[global_xyz[0]-patch_size, global_xyz[1]-patch_size, global_xyz[0]+patch_size, global_xyz[1]+patch_size],
        figsize=(10, 10),
        layer_names=['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'lane', 'road_segment', 'road_block'],
        render_egoposes_range=False,
        render_legend=False,
        alpha=0.5
    )

    name = "{}-{}".format(metas["timestamp"], metas["token"])

    save_path = f'all_{args.split}_basemaps/{name}_base_map_image.png'
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    metas_save_path = f'all_{args.split}_metas/{name}_metas.npy'
    np.save(metas_save_path, metas)

    if args.mode == "gt" and "gt_masks_bev" in data:
        masks = data["gt_masks_bev"].data[0].numpy()
        masks = masks.astype(np.bool)
    else:
        masks = None

    if masks is not None:
        visualize_map(
            os.path.join(args.out_dir, "map", f"{rank}{name}_generated_map_image.png"),
            masks,
            classes=cfg.map_classes,
        )

def main():
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

    np.set_printoptions(suppress=True)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")
        model.cuda()
        model.eval()

    nusc = NuScenes(version='v1.0-trainval', dataroot='/data1/data/nuscenes/', verbose=True)

    # def init_worker():
    #     global nusc_map
    #     nusc_map = NuScenesMap(dataroot='/data1/data/nuscenes/', map_name='boston-seaport')

    num_workers=16
    with mp.Pool(num_workers) as pool:
        results = []
        for data in tqdm(dataflow):
            results.append(pool.apply_async(process_sample, (data, args, cfg, nusc)))
        
        for result in results:
            result.get()

if __name__ == "__main__":
    main()
