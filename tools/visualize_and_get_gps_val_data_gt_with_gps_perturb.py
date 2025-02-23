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
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    np.set_printoptions(suppress = True)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
    torch.cuda.set_device(dist.local_rank())

    # build the dataloader
    dataset = build_dataset(cfg.data[args.split])

    #Code for using a subset of the data

    from torch.utils.data import Subset
    start=5000
    end=6019
    subset_indices = list(range(start, end))
    subset_dataset = Subset(dataset, subset_indices)
    dataflow = build_dataloader(
        subset_dataset,
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
        # print("Data keys:",data.keys())
        # print("Data metas:",data["metas"].data[0][0].keys())
        # print(data["metas"].data[0][0]['lidar_path'])
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

        #Obtained coordinates are nearly correct.
        # print("Global coordinates:", global_xyz)
        # wgs84 = pyproj.Proj(proj="latlong", datum="WGS84")
        # utm = pyproj.Proj(proj="utm", zone=48, datum="WGS84")  # Adjust UTM zone if needed
        # lon, lat = pyproj.transform(utm, wgs84, global_xyz[0], global_xyz[1])

        # print("Latitude:", lat, "Longitude:", lon)

        # Define the map patch around the ego vehicle
        patch_size = 500  # 200 meters in each direction (1km total)
        
        # Perturb the global coordinates
        perturbation = np.random.randint(-1*(patch_size-100), (patch_size-100), size=2)
        global_xyz[0] += perturbation[0]
        global_xyz[1] += perturbation[1]

        metas['perturbation'] = perturbation
        metas['new_global_xyz'] = global_xyz

        dpi = 200  # Set an appropriate DPI value
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


        save_path = 'all_val_basemaps_v2/'+name+"_base_map_image.png"  # Change this to your desired save location
        image = Image.fromarray(map_array)

        # Save the image to a file
        image.save(save_path)
        # print(f"Map image saved at: {save_path}")
        
        #Save metas variable to file
        #print("map_array shape:", map_array.shape)
        metas_save_path = 'all_val_metas_v2/'+name+"_metas.npy"  # Change this to your desired save location
        np.save(metas_save_path, metas)


        # if args.mode == "pred":
        #     with torch.inference_mode():
        #         outputs = model(**data)

        # # if args.mode == "gt" and "gt_bboxes_3d" in data:
        # #     bboxes = data["gt_bboxes_3d"].data[0][0].tensor.numpy()
        # #     labels = data["gt_labels_3d"].data[0][0].numpy()

        # #     if args.bbox_classes is not None:
        # #         indices = np.isin(labels, args.bbox_classes)
        # #         bboxes = bboxes[indices]
        # #         labels = labels[indices]

        # #     bboxes[..., 2] -= bboxes[..., 5] / 2
        # #     # print("GT",bboxes)
        # #     bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        # # elif args.mode == "pred" and "boxes_3d" in outputs[0]:
        # #     bboxes = outputs[0]["boxes_3d"].tensor.numpy()
        # #     scores = outputs[0]["scores_3d"].numpy()
        # #     labels = outputs[0]["labels_3d"].numpy()

        # #     if args.bbox_classes is not None:
        # #         indices = np.isin(labels, args.bbox_classes)
        # #         bboxes = bboxes[indices]
        # #         scores = scores[indices]
        # #         labels = labels[indices]

        # #     if args.bbox_score is not None:
        # #         indices = scores >= args.bbox_score
        # #         bboxes = bboxes[indices]
        # #         scores = scores[indices]
        # #         labels = labels[indices]

        # #     bboxes[..., 2] -= bboxes[..., 5] / 2
        # #     # print("PRED",bboxes)
        # #     bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        # # else:
        # #     bboxes = None
        # #     labels = None

        if args.mode == "gt" and "gt_masks_bev" in data:
            masks = data["gt_masks_bev"].data[0].numpy()
            masks = masks.astype(np.bool)
        # elif args.mode == "pred" and "masks_bev" in outputs[0]:
        #     masks = outputs[0]["masks_bev"].numpy()
        #     masks = masks >= args.map_score
        else:
            masks = None

        # # if "img" in data:
        # #     for k, image_path in enumerate(metas["filename"]):
        # #         image = mmcv.imread(image_path)
        # #         #print("Debuggging inside visuliase.py ******* ")
        # #         #print(k)
        # #         #print(image_path)
        # #         #print(metas["lidar2image"][k])

        # #         visualize_camera(
        # #             os.path.join(args.out_dir, f"camera-{k}", f"{name}.png"),
        # #             image,
        # #             bboxes=bboxes,
        # #             labels=labels,
        # #             transform=metas["lidar2image"][k],
        # #             classes=cfg.object_classes,
        # #         )

        # # if "points" in data:
        # #     lidar = data["points"].data[0][0].numpy()
        # #     visualize_lidar(
        # #         os.path.join(args.out_dir, "lidar", f"{name}.png"),
        # #         lidar,
        # #         bboxes=bboxes,
        # #         labels=labels,
        # #         xlim=[cfg.point_cloud_range[d] for d in [0, 3]],
        # #         ylim=[cfg.point_cloud_range[d] for d in [1, 4]],
        # #         classes=cfg.object_classes,
        # #     )

        if masks is not None:
            # print("Saving generated map")
            # print(os.path.join(args.out_dir, "map", f"{name}.png"))
            visualize_map(
                os.path.join(args.out_dir, "map", f"{name}_generated_map_image.png"),
                masks,
                classes=cfg.map_classes,
            )


if __name__ == "__main__":
    main()
