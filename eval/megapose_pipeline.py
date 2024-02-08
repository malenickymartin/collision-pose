# Standard Library
import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Union

# Third Party
import numpy as np
import torch
from bokeh.io import export_png
from bokeh.plotting import gridplot
from PIL import Image

# HappyPose
from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData
from happypose.toolbox.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from happypose.toolbox.inference.utils import make_detections_from_object_data
from happypose.toolbox.lib3d.transform import Transform
from happypose.toolbox.renderer import Panda3dLightData
from happypose.toolbox.renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from happypose.toolbox.utils.conversion import convert_scene_observation_to_panda3d
from happypose.toolbox.utils.load_model import NAMED_MODELS, load_named_model
from happypose.toolbox.utils.logging import get_logger, set_logging_level
from happypose.toolbox.visualization.bokeh_plotter import BokehPlotter
from happypose.toolbox.visualization.utils import make_contour_overlay

# MegaPose
# from happypose.toolbox.datasets.object_dataset import RigidObject, RigidObjectDataset
# from happypose.toolbox.datasets.scene_dataset import CameraData, ObjectData


logger = get_logger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_observation(
    example_dir: Path,
    idx: int,
    load_depth: bool = False,
) -> Tuple[np.ndarray, Union[None, np.ndarray], CameraData]:
    with open(example_dir / "train_pbr/000000/scene_camera.json", "rb") as f:
        json_data = json.load(f)
    K = json_data[str(idx)]["cam_K"]
    K_new = [K[:3], K[3:6], K[6:9]]
    json_data = {"K": K_new, "resolution": [480, 640]}
    json_data = json.dumps(json_data)
    camera_data = CameraData.from_json(json_data)

    rgb = np.array(Image.open(example_dir / f"train_pbr/000000/rgb/{idx:06d}.jpg"), dtype=np.uint8)
    assert rgb.shape[:2] == camera_data.resolution

    depth = None
    if load_depth:
        depth = (
            np.array(
                Image.open(example_dir / f"train_pbr/000000/depth/{idx:06d}.png"),
                dtype=np.float32,
            )
            / 1000
        )
        assert depth.shape[:2] == camera_data.resolution

    return rgb, depth, camera_data


def load_observation_tensor(
    example_dir: Path,
    idx: int,
    load_depth: bool = False,
) -> ObservationTensor:
    rgb, depth, camera_data = load_observation(example_dir, idx, load_depth)
    observation = ObservationTensor.from_numpy(rgb, depth, camera_data.K)
    return observation


def load_output_data(data_path: Path) -> List[ObjectData]:
    object_data = json.loads(data_path.read_text())
    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data


def load_object_data(data_path: Path, idx: int) -> List[ObjectData]:
    with open(data_path / "train_pbr/000000/scene_gt.json", "r") as f:
        scene_gt = json.load(f)
    with open(data_path / "train_pbr/000000/scene_gt_info.json", "r") as f:
        scene_gt_info = json.load(f)

    labels = []
    bboxes = []
    visib_fract = []
    for obj in range(len(scene_gt[str(idx)])):
        if scene_gt_info[str(idx)][obj]["px_count_visib"] > 300:
            labels.append(scene_gt[str(idx)][obj]["obj_id"])
            bbox = scene_gt_info[str(idx)][obj]["bbox_visib"]
            bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
            dw = 5 * (bbox[2] - bbox[0]) / 100
            dh = 5 * (bbox[3] - bbox[1]) / 100
            bbox[0] = np.clip(bbox[0] - dw, 0, 640 + 1)
            bbox[1] = np.clip(bbox[1] - dh, 0, 480 + 1)
            bbox[2] = np.clip(bbox[2] + dw, 0, 640 - 1)
            bbox[3] = np.clip(bbox[3] + dh, 0, 480 - 1)
            bboxes.append(bbox)
            visib_fract.append(scene_gt_info[str(idx)][obj]["visib_fract"])

    object_data = []
    for i in range(len(labels)):
        object_data.append(
            {
                "label": str(labels[i]),
                "obj_id": str(labels[i]),
                "bbox_modal": bboxes[i],
                "visib_fract": visib_fract[i],
            }
        )

    object_data = [ObjectData.from_json(d) for d in object_data]
    return object_data


def load_detections(example_dir: Path, idx: int) -> DetectionsType:
    input_object_data = load_object_data(example_dir, idx)
    detections = make_detections_from_object_data(input_object_data).to(device)
    return detections


def make_object_dataset(example_dir: Path) -> RigidObjectDataset:
    rigid_objects = []
    mesh_units = "mm"
    object_dirs = (example_dir / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
        # TODO: fix mesh units
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset


def make_detections_visualization(example_dir: Path, idx: int) -> None:
    rgb, _, _ = load_observation(example_dir, idx, load_depth=False)
    detections = load_detections(example_dir, idx)
    plotter = BokehPlotter()
    fig_rgb = plotter.plot_image(rgb)
    fig_det = plotter.plot_detections(fig_rgb, detections=detections)
    output_fn = example_dir / "happypose" / "detections" / f"detection_{idx}.png"
    output_fn.parent.mkdir(exist_ok=True, parents=True)
    export_png(fig_det, filename=output_fn)
    logger.info(f"Wrote detections visualization: {output_fn}")
    return


def save_predictions(
    example_dir: Path,
    idx: int,
    pose_estimates: PoseEstimatesType,
) -> None:
    labels = pose_estimates.infos["label"]
    poses = pose_estimates.poses.cpu().numpy()
    object_data = [
        ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
    ]
    object_data_json = json.dumps([x.to_json() for x in object_data])
    output_fn = example_dir / "happypose" / "outputs" / f"object_data_{idx}.json"
    output_fn.parent.mkdir(exist_ok=True)
    output_fn.write_text(object_data_json)
    logger.info(f"Wrote predictions: {output_fn}")
    return


def make_output_visualization(example_dir: Path, idx: int, object_dataset) -> None:
    rgb, _, camera_data = load_observation(example_dir, idx, load_depth=False)
    camera_data.TWC = Transform(np.eye(4))
    object_datas = load_output_data(
        example_dir / "happypose" / "outputs" / f"object_data_{idx}.json"
    )

    renderer = Panda3dSceneRenderer(object_dataset)

    camera_data, object_datas = convert_scene_observation_to_panda3d(camera_data, object_datas)
    light_datas = [
        Panda3dLightData(
            light_type="ambient",
            color=((1.0, 1.0, 1.0, 1)),
        ),
    ]
    renderings = renderer.render_scene(
        object_datas,
        [camera_data],
        light_datas,
        render_depth=False,
        render_binary_mask=False,
        render_normals=False,
        copy_arrays=True,
    )[0]

    plotter = BokehPlotter()

    fig_rgb = plotter.plot_image(rgb)
    fig_mesh_overlay = plotter.plot_overlay(rgb, renderings.rgb)
    contour_overlay = make_contour_overlay(
        rgb, renderings.rgb, dilate_iterations=1, color=(0, 255, 0)
    )["img"]
    fig_contour_overlay = plotter.plot_image(contour_overlay)
    fig_all = gridplot([[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None)
    vis_dir = example_dir / "happypose" / "visualizations"
    vis_dir.mkdir(exist_ok=True, parents=True)
    export_png(fig_all, filename=vis_dir / f"all_results_{idx}.png")
    logger.info(f"Wrote visualizations to {vis_dir}.")
    return


def run_inference(
    example_dir: Path,
    model_name: str,
) -> None:
    logger.info(f"Loading model {model_name}.")
    model_info = NAMED_MODELS[model_name]
    object_dataset = make_object_dataset(example_dir)
    pose_estimator = load_named_model(model_name, object_dataset, bsz_images=64).to(device)
    logger.info(f"Running inference.")


    for idx in range(len(list((example_dir/"train_pbr/000000/rgb").iterdir()))):
        make_detections_visualization(example_dir, idx)

        observation = load_observation_tensor(
            example_dir, idx, load_depth=model_info["requires_depth"]
        )
        if torch.cuda.is_available():
            observation.cuda()

        detections = load_detections(example_dir, idx).to(device)

        output, _ = pose_estimator.run_inference_pipeline(
            observation,
            detections=detections,
            **model_info["inference_parameters"],
        )
        save_predictions(example_dir, idx, output)

        make_output_visualization(example_dir, idx, object_dataset)

    return


if __name__ == "__main__":
    set_logging_level("info")
    parser = argparse.ArgumentParser()
    parser.add_argument("example_name", nargs="?", default="ycbv_convex_two")
    parser.add_argument("--model", type=str, default="megapose-1.0-RGB-multi-hypothesis")
    parser.add_argument("--vis-detections", action="store_true", default=True)
    parser.add_argument("--run-inference", action="store_true", default=True)
    parser.add_argument("--vis-outputs", action="store_true", default=True)
    args = parser.parse_args()

    data_dir = Path("/local2/homes/malenma3/object_detection/datasets") / args.example_name
    assert data_dir

    run_inference(data_dir, args.model)
