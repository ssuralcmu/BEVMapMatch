import argparse
import copy
import os
import pyproj
import mmcv
import numpy as np
import torch
import torch.multiprocessing as mp
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from tqdm import tqdm
from torchpack.utils.config import configs
from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.core.utils import visualize_camera, visualize_lidar, visualize_map
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
import matplotlib.pyplot as plt
import io
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg

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
    fig, ax = nusc_map.render_map_patch(*args, **kwargs)

    bbox_coords = kwargs.get('box_coords')
    ax.set_xlim(bbox_coords[0], bbox_coords[2])
    ax.set_ylim(bbox_coords[1], bbox_coords[3])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.axis('off')

    fig.set_dpi(dpi)
    
    buf = io.BytesIO()
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(buf)
    
    fig.clf()
    
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)

def worker(rank, world_size, args, cfg):
    torch.manual_seed(rank)
    np.random.seed(rank)

    print(f"Worker {rank} started")
    print(cfg)

    dataset = build_dataset(cfg.data[args.split])
    dataflow = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        num_gpus=1,
        dist=False,
        shuffle=False,
    )

    if args.mode == "pred":
        model = build_model(cfg.model)
        load_checkpoint(model, args.checkpoint, map_location="cpu")
        model = MMDataParallel(model, device_ids=[0])
        model.eval()

    from nuscenes.nuscenes import NuScenes
    from nuscenes.map_expansion.map_api import NuScenesMap

    nusc = NuScenes(version='v1.0-trainval', dataroot='/data1/data/nuscenes/', verbose=True)

    for i, data in enumerate(dataflow):
        if i % world_size != rank:
            continue

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
        perturbation = np.random.randint(-1*(patch_size-100), (patch_size-100), size=2)
        global_xyz[0] += perturbation[0]
        global_xyz[1] += perturbation[1]

        metas['perturbation'] = perturbation
        metas['new_global_xyz'] = global_xyz

        dpi = 200
        figsize = (2*2*patch_size/dpi, 2*2*patch_size/dpi)

        map_array = render_map_patch_to_array(nusc_map,
            box_coords=[global_xyz[0]-patch_size, global_xyz[1]-patch_size, global_xyz[0]+patch_size, global_xyz[1]+patch_size],
            figsize=figsize,
            layer_names=['drivable_area', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area', 'lane', 'road_segment', 'road_block'],
            render_egoposes_range=False,
            render_legend=False,
            alpha=0.5,
            dpi=dpi
        )

        name = "{}-{}".format(metas["timestamp"], metas["token"])

        save_path = f'all_val_basemaps_v2/{name}_base_map_image.png'
        image = Image.fromarray(map_array)
        image.save(save_path)

        metas_save_path = f'all_val_metas_v2/{name}_metas.npy'
        np.save(metas_save_path, metas)

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
        else:
            masks = None

        if masks is not None:
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}_generated_map_image.png"),
                masks,
                classes=cfg.map_classes,
            )



if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

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
        parser.add_argument("--num-workers", type=int, default=16)
        args,opts = parser.parse_known_args()
    
        configs.load(args.config, recursive=True)
        configs.update(opts)

        cfg = Config(recursive_eval(configs), filename=args.config)

        processes = []
        for rank in range(args.num_workers):
            p = mp.Process(target=worker, args=(rank, args.num_workers, args, cfg))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    main()
