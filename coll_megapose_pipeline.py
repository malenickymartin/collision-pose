# Standard Library
import json
from pathlib import Path
from typing import List, Union, Dict, Tuple

# Third Party
import pinocchio as pin
import hppfcl
import numpy as np
from PIL import Image
import torch
from bokeh.io import export_png
from bokeh.plotting import gridplot

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
from happypose.toolbox.visualization.bokeh_plotter import BokehPlotter
from happypose.toolbox.visualization.utils import make_contour_overlay

# CollisionPose
from run_optim import optim
from scene import DiffColScene, SelectStrategyConfig
import pydiffcol
from pydiffcol.utils import select_strategy
from eval.eval_utils import get_se3_from_mp_json, get_se3_from_bp_cam

class CollisionResolver:
    """
    Class to run optimization using the DiffCol library

    Inputs:
    - meshes_dir: Path to directory containing directories with object meshes
    - meshes_decomp_dir: Path to directory containing directories with decomposed object meshes or None
    - static_meshes_dir: Path to file with static object or Path to directory containing directories with static object meshes or None
    - static_meshes_decomp_dir: Path to directory containing directories with decomposed static object meshes or None
    - params: dict with optimization parameters or None    
    """
    def __init__(self, meshes_dir: Path, meshes_decomp_dir: Union[Path, None] = None,
                 static_meshes_dir: Union[Path, None] = None, static_meshes_decomp_dir: Union[Path, None] = None,
                 params: Union[Dict[str, Union[str,int,List]], None] = None):
        self.args = SelectStrategyConfig(1e-2, 100)
        self.col_req, self.col_req_diff = select_strategy(self.args)
        self.params = params
        self.meshes, self.meshes_decomp, self.stat_meshes, self.stat_meshes_decomp = self.load_meshes(meshes_dir, meshes_decomp_dir, static_meshes_dir, static_meshes_decomp_dir)
        

    def create_mesh(self, mesh_loader: hppfcl.MeshLoader, obj_path: str) -> hppfcl.Convex:
        mesh = mesh_loader.load(obj_path, np.array(3*[0.001]))
        mesh.buildConvexHull(True, "Qt")
        return mesh.convex


    def create_decomposed_mesh(self, mesh_loader: hppfcl.MeshLoader, dir_path: str) -> List[hppfcl.Convex]:
        meshes = []
        for path in Path(dir_path).iterdir():
            if path.suffix == ".ply" or path.suffix == ".obj":
                mesh = mesh_loader.load(str(path), scale=np.array(3*[0.001]))
                mesh.buildConvexHull(True, "Qt")
                meshes.append(mesh.convex)
        return meshes
    

    def load_meshes(
            self, meshes_dir: Path, meshes_decomp_dir: Union[Path, None],
            static_meshes_dir: Union[Path, None], static_meshes_decomp_dir: Union[Path, None]
        ) -> Tuple[Dict[str, hppfcl.Convex], Dict[str, List[hppfcl.Convex]], Dict[str, hppfcl.Convex], Dict[str, List[hppfcl.Convex]]]:
        """
        Load meshes from the input directories

        Inputs:
        - meshes_dir: Path to directory containing directories with object meshes
        - meshes_decomp_dir: Path to directory containing directories with decomposed object meshes or None
        - static_meshes_dir: Path to file with static objet or Path to directory containing directories with static object meshes or None
        - static_meshes_decomp_dir: Path to directory containing directories with decomposed static object meshes or None

        Returns:
        - mesh_objs_dict: dict with keys as object labels (str) and values as fcl.Convex objects
        - mesh_objs_dict_decomposed: dict with keys as object labels (str) and values as list of fcl.Convex objects
        - mesh_stat_dict: dict with keys as object labels (str) and values as fcl.Convex objects
        - mesh_stat_dict_decomposed: dict with keys as object labels (str) and values as list of fcl.Convex objects
        """
        mesh_loader = hppfcl.MeshLoader()

        print("Loading meshes...")
        mesh_objs_dict = {}
        for mesh_dir_path in meshes_dir.iterdir():
            mesh_label = str(mesh_dir_path.name)
            mesh_path = mesh_dir_path / f"obj_{int(mesh_label):06d}.ply"
            mesh_objs_dict[mesh_label] = self.create_mesh(mesh_loader, str(mesh_path))

        mesh_objs_dict_decomposed = {}
        if meshes_decomp_dir is not None:
            print("Loading decomposed meshes...")
            for mesh_dir_path in meshes_decomp_dir.iterdir():
                mesh_label = str(mesh_dir_path.name)
                mesh_objs_dict_decomposed[mesh_label] = self.create_decomposed_mesh(mesh_loader, str(mesh_dir_path))

        mesh_stat_dict = {}
        if static_meshes_dir is not None:
            print("Loading static meshes...")
            if static_meshes_dir.is_file():
                mesh_stat_dict[static_meshes_dir.stem] = self.create_mesh(mesh_loader, str(static_meshes_dir))
            else:
                for mesh_dir_path in static_meshes_decomp_dir.iterdir():
                    mesh_label = str(mesh_dir_path.name)
                    mesh_path = mesh_dir_path / f"obj_{int(mesh_label):06d}.ply"
                    mesh_stat_dict[mesh_label] = self.create_mesh(mesh_loader, str(mesh_path))
                
        mesh_stat_dict_decomposed = {}
        if static_meshes_decomp_dir is not None:
            print("Loading decomposed static meshes...")
            for mesh_dir_path in static_meshes_decomp_dir.iterdir():
                mesh_label = str(mesh_dir_path.name)
                mesh_stat_dict_decomposed[mesh_label] = self.create_decomposed_mesh(mesh_loader, str(mesh_dir_path))

        return mesh_objs_dict, mesh_objs_dict_decomposed, mesh_stat_dict, mesh_stat_dict_decomposed


    def run_optim_inference(self, poses: Dict[str, pin.SE3], static_poses: Union[Dict[str, pin.SE3], None] = None,
                            save_path: Path = None, visualize: bool = False) -> Dict[str, pin.SE3]:
        """
        Run optimization on the input poses and static poses using the pre-loaded meshes

        Inputs:
        - poses: dict with keys as object labels (str) and values as SE3 poses (pin.SE3)
        - static_poses: dict with keys as object labels (str) and values as SE3 poses (pin.SE3) or None
        - save_path: Path to save the output json file or None
        - visualize: bool to visualize the optimization process

        Returns:
        - final_poses: dict with keys as object labels (str) and values as SE3 poses (pin.SE3)
        """
        curr_meshes = []
        curr_meshes_decomp = []
        wMo_lst = []
        labels = list(poses.keys())
        for label in labels:
            curr_meshes.append(self.meshes[label])
            if len(self.meshes_decomp) > 0:
                curr_meshes_decomp.append(self.meshes_decomp[label])
            wMo_lst.append(poses[label])
        curr_static_meshes = []
        curr_static_meshes_decomp = []
        wMs_lst = []
        if static_poses is not None:
            static_labels = list(static_poses.keys())
            for label in static_labels:
                curr_static_meshes.append(self.stat_meshes[label])
                if len(self.stat_meshes_decomp) > 0:
                    curr_static_meshes_decomp.append(self.stat_meshes_decomp[label])
                wMs_lst.append(static_poses[label])
        dc_scene = DiffColScene(curr_meshes, curr_static_meshes, wMs_lst, curr_meshes_decomp, curr_static_meshes_decomp, pre_loaded_meshes=True)
        X = optim(dc_scene, wMo_lst, self.col_req, self.col_req_diff, self.params, visualize)

        final_poses = {}
        for se3, label in zip(X, labels):
            final_poses[label] = se3

        if save_path != None:
            assert save_path.suffix == ".json", "Save path must be a json file"
            to_json = []
            for se3, label in zip(X, labels):
                xyzquat = pin.SE3ToXYZQUAT(se3)
                json_dict = {"label":label, "TWO":[list(xyzquat[3:]), list(xyzquat[:3])]}
                to_json.append(json_dict)
            save_path.write_text(json.dumps(to_json))
        
        return final_poses


class MegaposePredictor:
    """
    Class to run inference using the Megapose model from HappyPose
    
    Inputs:
    - meshes_dir: Path to directory containing object meshes
    - model_name: Name of the Megapose model to use
    - camera_data: dict with camera intrinsics {"K":3x3 list, "resolution": [H,W]} or None (will be different for each image)
    """
    def __init__(
        self,
        meshes_dir: Path,
        model_name: str = "megapose-1.0-RGB",
        camera_data: Union[Dict[str, List], None] = None,
    ):
        self.object_dataset = self.make_object_dataset(meshes_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model {model_name}.")
        self.model_info = NAMED_MODELS[model_name]
        self.pose_estimator = load_named_model(model_name, self.object_dataset).to(self.device)
        if camera_data == None:
            self.camera_data = None
        else:
            self.camera_data = CameraData.from_json(json.dumps(camera_data))


    def load_output_data(self, data_path: Path) -> List[ObjectData]:
        object_data = json.loads(data_path.read_text())
        object_data = [ObjectData.from_json(d) for d in object_data]
        return object_data


    def make_object_dataset(self, models_dir: Path) -> RigidObjectDataset:
        rigid_objects = []
        mesh_units = "mm"
        object_dirs = models_dir.iterdir()
        for object_dir in object_dirs:
            label = object_dir.name
            mesh_path = None
            for fn in object_dir.glob("*"):
                if fn.suffix in {".obj", ".ply"}:
                    assert not mesh_path, f"there multiple meshes in the {label} directory"
                    mesh_path = fn
            assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
            rigid_objects.append(
                RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units)
            )
        rigid_object_dataset = RigidObjectDataset(rigid_objects)
        return rigid_object_dataset


    def make_detections_visualization(self, rgb: np.ndarray, detections: List[Dict[str, Union[str, List[int], int, float]]], save_dir: Path) -> None:
        plotter = BokehPlotter()
        fig_rgb = plotter.plot_image(rgb)
        fig_det = plotter.plot_detections(fig_rgb, detections=detections)
        output_fn = save_dir / "visualizations" / "detection.png"
        output_fn.parent.mkdir(exist_ok=True, parents=True)
        export_png(fig_det, filename=output_fn)
        print(f"Wrote detections visualization: {output_fn}")


    def save_predictions(
        self,
        save_dir: Path,
        pose_estimates: PoseEstimatesType,
    ) -> None:
        labels = pose_estimates.infos["label"]
        poses = pose_estimates.poses.cpu().numpy()
        object_data = [
            ObjectData(label=label, TWO=Transform(pose)) for label, pose in zip(labels, poses)
        ]
        object_data_json = json.dumps([x.to_json() for x in object_data])
        output_fn = save_dir / "object_data.json"
        output_fn.parent.mkdir(exist_ok=True)
        output_fn.write_text(object_data_json)
        print(f"Wrote predictions: {output_fn}")


    def make_output_visualization(self, rgb: np.ndarray, camera_data: CameraData, save_dir: Path, object_dataset: RigidObjectDataset) -> None:
        camera_data.TWC = Transform(np.eye(4))
        object_datas = self.load_output_data(save_dir / "object_data.json")

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
        fig_all = gridplot(
            [[fig_rgb, fig_contour_overlay, fig_mesh_overlay]], toolbar_location=None
        )
        vis_dir = save_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True, parents=True)
        export_png(fig_all, filename=vis_dir / "all_results.png")
        print(f"Wrote visualizations to {vis_dir}.")


    def run_megapose_inference(
        self,
        rgb: np.ndarray,
        detections: List[Dict[str, Union[str, List[int], int, float]]],
        depth: Union[np.ndarray, None] = None,
        camera_data: Union[dict, None] = None,
        save_dir: Union[Path, None] = None,
    ) -> Dict[str, pin.SE3]:
        """
        Run inference on a single image

        Inputs:
        - rgb: np.ndarray of shape (H,W,3)
        - detections: list of dicts with keys "label", "bbox_modal", "obj_id", "visib_fract"
        - depth: np.ndarray of shape (H,W) or None
        - camera_data: dict with camera intrinsics {"K":3x3 np.ndarray, "resolution": [H,W]} or None
        - save_dir: Path to save visualizations or None
        
        Returns:
        - object_data: dict with keys "label" and values np.ndarray of shape (7,)
        """
        assert camera_data != None or self.camera_data != None, "Camera data is required for inference"
        if camera_data == None:
            camera_data = self.camera_data
        else:
            camera_data = CameraData.from_json(json.dumps(camera_data))
        assert camera_data.resolution == rgb.shape[:2], "Camera resolution is not the same as image resolution"

        assert not self.model_info["requires_depth"] or depth is not None, "Depth is required for this model"
        observation = ObservationTensor.from_numpy(rgb, depth, self.camera_data.K)

        if self.device != "cpu":
            observation.cuda()

        assert len(detections) > 0, "No detections provided"
        input_object_data = [ObjectData.from_json(d) for d in detections]
        detections = make_detections_from_object_data(input_object_data).to(self.device)

        print("Inference running...")
        output, _ = self.pose_estimator.run_inference_pipeline(
            observation,
            detections=detections,
            **self.model_info["inference_parameters"],
        )
        print("Inference complete")

        if save_dir != None:
            self.make_detections_visualization(rgb, detections, save_dir)
            self.save_predictions(save_dir, output)
            self.make_output_visualization(rgb, self.camera_data, save_dir, self.object_dataset)

        labels = output.infos["label"]
        poses = output.poses.cpu().numpy()
        object_data = {label:pin.SE3(pose) for label, pose in zip(labels, poses)}
        return object_data


def example():
    data_dir = Path("/local2/homes/malenma3/collision-pose/eval/data/datasets") / "ycbv_convex_one"

    # MEGAPOSE CLASS
    meshes_dir = data_dir / "meshes"
    camera_data = {
        "K": [[647.3, 0.0, 320.0], [0.0, 647.3, 240.0], [0.0, 0.0, 1.0]],
        "resolution": [480, 640]
    }
    megapose = MegaposePredictor(meshes_dir, "megapose-1.0-RGB", camera_data)
    # MEGAPOSE INFERENCE
    rgb = np.array(Image.open(data_dir / f"train_pbr/000000/rgb/000000.jpg"), dtype=np.uint8)
    detections = [{"bbox_modal": [234, 157, 234+171, 157+164], "visib_fract": 1.0, "label": "1"}]
    save_dir = data_dir / "test_output"
    poses = megapose.run_megapose_inference(rgb, detections, save_dir=save_dir)
    # DIFFCOL CLASS
    meshes_dir = data_dir / "meshes"
    meshes_decomp_dir = data_dir / "meshes_decomp"
    static_meshes_dir = Path("eval/data/floor.ply")
    diffcol = CollisionResolver(meshes_dir, meshes_decomp_dir, static_meshes_dir)
    # DIFFCOL INFERENCE
    static_pose = pin.SE3(np.eye(4))
    static_pose.translation = np.array([-82.9, -9.8, 700.4]) / 1000
    static_pose.rotation = np.array([-0.787, -0.254, 0.562,
                                     -0.617, 0.335, -0.712,
                                     -0.007, -0.907, -0.42]).reshape(3, 3)
    static_poses = {"floor": static_pose}
    poses_optimized = diffcol.run_optim_inference(poses, static_poses, save_path=save_dir / "object_data_optimized.json", visualize=True)
    # RESULTS
    print(poses_optimized)
    errors = [np.linalg.norm(pin.log6(poses[label]) - pin.log6(poses_optimized[label])) for label in poses.keys()]
    print(errors)
    return poses_optimized



if __name__ == "__main__":
    example()