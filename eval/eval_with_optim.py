import time
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import pinocchio as pin
from pathlib import Path
import json
import hppfcl

import pydiffcol
from pydiffcol.utils import (
    select_strategy
)

from scene import DiffColScene, SelectStrategyConfig
from run_optim import optim

from eval.eval_utils import get_se3_from_mp_json, get_se3_from_bp_cam


def create_mesh(mesh_loader, obj_path: str):
    mesh = mesh_loader.load(obj_path, np.array(3*[0.001]))
    mesh.buildConvexHull(True, "Qt")
    return mesh.convex

def create_decomposed_mesh(mesh_loader, dir_path: str):
    meshes = []
    for path in Path(dir_path).iterdir():
        if path.suffix == ".ply" or path.suffix == ".obj":
            mesh = mesh_loader.load(str(path), scale=np.array(3*[0.001]))
            mesh.buildConvexHull(True, "Qt")
            meshes.append(mesh.convex)
    return meshes

if __name__ == "__main__":

    visualize = False
    N_step = 500

    args = SelectStrategyConfig()
    args.noise = 1e-2
    args.num_samples = 100
    col_req, col_req_diff = select_strategy(args)

    path_stat_objs = ["eval/data/floor.ply"]

    print("Loading decomposed meshes...")
    mesh_loader = hppfcl.MeshLoader()
    path_objs_all = Path("eval/data/ycbv_convex")
    path_objs_decomposed = path_objs_all / "meshes_decomp"
    mesh_objs_dict_decomposed = {}
    for mesh_dir_path in path_objs_decomposed.iterdir():
        mesh_label = int(mesh_dir_path.name)
        mesh_objs_dict_decomposed[mesh_label] = create_decomposed_mesh(mesh_loader, str(mesh_dir_path))

    print("Loading meshes...")
    path_objs_convex = path_objs_all / "meshes"
    mesh_objs_dict = {}
    for mesh_dir_path in path_objs_convex.iterdir():
        mesh_label = int(mesh_dir_path.name)
        mesh_path = mesh_dir_path / f"obj_{mesh_label:06d}.ply"
        mesh_objs_dict[mesh_label] = create_mesh(mesh_loader, str(mesh_path))
    static_meshes = [create_mesh(mesh_loader, str(p)) for p in path_stat_objs]
    
    path_wMo_all = Path("eval/data/ycbv_convex/happypose/outputs")
    gt_cam_path = Path("eval/data/ycbv_convex/train_pbr/000000/scene_camera.json")
    wMo_lst_all = []
    wMs_lst = []
    label_objs_all = []
    scene_idxs = []
    with open(gt_cam_path, "r") as f:
        gt_cam = json.load(f)
    for wMo_path in path_wMo_all.iterdir():
        scene_idx = int(wMo_path.name.split("_")[-1].split(".")[0])
        with open(wMo_path, "r") as f:
            wMo_json = json.load(f)
        wMo_lst = []
        label_objs = []
        se3_cam = get_se3_from_bp_cam(gt_cam[str(scene_idx)])
        for i in range(len(wMo_json)):
            label = int(wMo_json[i]["label"])
            wMo = get_se3_from_mp_json(wMo_json[i])
            wMo_lst.append(wMo)
            label_objs.append(label)
        wMo_lst_all.append(wMo_lst)
        wMs_lst.append(se3_cam)
        label_objs_all.append(label_objs)
        scene_idxs.append(scene_idx)

    save_dir_path = Path("eval/data/ycbv_convex/happypose/outputs_coll")
    for i in tqdm(range(len(wMo_lst_all))):
        curr_meshes = []
        curr_meshes_decomp = []
        for l in label_objs_all[i]:
            curr_meshes.append(mesh_objs_dict[l])
            curr_meshes_decomp.append(mesh_objs_dict_decomposed[l])
        wMo_lst = wMo_lst_all[i]
        dc_scene = DiffColScene(curr_meshes, static_meshes, [wMs_lst[i]], curr_meshes_decomp, pre_loaded_meshes=True)
        X = optim(dc_scene, wMo_lst, col_req, col_req_diff, N_step, "adam", visualize)
        to_json = []
        for j in range(len(X)):
            xyzquat = pin.SE3ToXYZQUAT(X[j])
            json_dict = {"label":str(label_objs_all[i][j]), "TWO":[list(xyzquat[3:]), list(xyzquat[:3])]}
            to_json.append(json_dict)
        save_data_path = save_dir_path / f"object_data_{scene_idxs[i]}.json"
        save_data_path.write_text(json.dumps(to_json))