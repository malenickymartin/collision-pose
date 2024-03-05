import time
from copy import deepcopy
from tqdm import tqdm
import sys
import numpy as np
import pinocchio as pin
from pathlib import Path
import json
import hppfcl
from typing import Union, List, Dict

import pydiffcol
from pydiffcol.utils import (
    select_strategy
)

from scene import DiffColScene, SelectStrategyConfig
from run_optim import optim
from config import MESHES_PATH, MESHES_DECOMP_PATH, FLOOR_MESH_PATH, DATASETS_PATH, POSES_OUTPUT_PATH, FLOOR_POSES_PATH

from eval.eval_utils import get_se3_from_mp_json, get_se3_from_bp_cam, load_meshes, load_multi_convex_meshes, load_csv


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

def save_optimized_bproc(params):

    visualize = True
    dataset_name = "ycbv_convex"

    args = SelectStrategyConfig(1e-2, 100)
    col_req, col_req_diff = select_strategy(args)

    path_stat_objs = ["eval/data/floor.ply"]
    mesh_loader = hppfcl.MeshLoader()

    print("Loading decomposed meshes...")
    dataset_path = Path("eval/data") / dataset_name
    path_objs_decomposed = dataset_path / "meshes_decomp"
    mesh_objs_dict_decomposed = {}
    for mesh_dir_path in path_objs_decomposed.iterdir():
        mesh_label = int(mesh_dir_path.name)
        mesh_objs_dict_decomposed[mesh_label] = create_decomposed_mesh(mesh_loader, str(mesh_dir_path))

    print("Loading meshes...")
    path_objs_convex = dataset_path / "meshes"
    mesh_objs_dict = {}
    for mesh_dir_path in path_objs_convex.iterdir():
        mesh_label = int(mesh_dir_path.name)
        mesh_path = mesh_dir_path / f"obj_{mesh_label:06d}.ply"
        mesh_objs_dict[mesh_label] = create_mesh(mesh_loader, str(mesh_path))
    static_meshes = [create_mesh(mesh_loader, str(p)) for p in path_stat_objs]
    
    path_wMo_all = dataset_path / "happypose/outputs"
    gt_cam_path = dataset_path / "train_pbr/000000/scene_camera.json"
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

    save_dir_path = dataset_path / "happypose/outputs_coll"
    for i in tqdm(range(len(wMo_lst_all))):
        curr_meshes = []
        curr_meshes_decomp = []
        for l in label_objs_all[i]:
            curr_meshes.append(mesh_objs_dict[l])
            curr_meshes_decomp.append(mesh_objs_dict_decomposed[l])
        wMo_lst = wMo_lst_all[i]
        dc_scene = DiffColScene(curr_meshes, static_meshes, [wMs_lst[i]], curr_meshes_decomp, pre_loaded_meshes=True)
        X = optim(dc_scene, wMo_lst, col_req, col_req_diff, params, visualize)
        to_json = []
        for j in range(len(X)):
            xyzquat = pin.SE3ToXYZQUAT(X[j])
            json_dict = {"label":str(label_objs_all[i][j]), "TWO":[list(xyzquat[3:]), list(xyzquat[:3])]}
            to_json.append(json_dict)
        save_data_path = save_dir_path / f"object_data_{scene_idxs[i]}.json"
        save_data_path.write_text(json.dumps(to_json))


def save_optimized_floor(params:dict = None, vis:bool = False):
    """Optimizes pose of floor in YCBV BOP dataset using collision detection."""

    data_path = Path("eval/data")
    rigid_objects = load_meshes(data_path / "meshes")
    rigid_objects_decomp = load_multi_convex_meshes(data_path / "meshes_decomp")
    floor_mesh, floor_se3s = load_static()
    args = SelectStrategyConfig(1e-2, 100)
    col_req, col_req_diff = select_strategy(args)

    optimized_floor = {}

    for scene in tqdm(floor_se3s):
        with open(Path("/local2/homes/malenma3/collision-pose/eval/data/ycbv_test_dataset") / f"{int(scene):06d}" / "scene_gt.json", "r") as f:
            gt_poses_all = json.load(f)
        optimized_floor[scene] = {}
        for im in tqdm(floor_se3s[scene]):
            curr_stat_meshes = []
            curr_stat_meshes_decomp = []
            wMs_lst = []
            # Load info about each object in the scene
            for obj in gt_poses_all[im]:
                wMs_lst.append(pin.SE3(np.array(obj["cam_R_m2c"]).reshape(3,3), np.array(obj["cam_t_m2c"])/1000))
                curr_stat_meshes.append(rigid_objects[str(obj["obj_id"])])
                curr_stat_meshes_decomp.append(rigid_objects_decomp[str(obj["obj_id"])])
            wMo = floor_se3s[str(scene)][str(im)]
            wMo_lst, curr_meshes = (None, None) if wMo is None else ([pin.SE3(np.array(wMo["R"]), np.array(wMo["t"]))], [floor_mesh])
            if wMo_lst is None:
                optimized_floor[scene][im] = None
                continue
            dc_scene = DiffColScene(curr_meshes, curr_stat_meshes, wMs_lst, [], curr_stat_meshes_decomp, pre_loaded_meshes=True)
            X = optim(dc_scene, wMo_lst, col_req, col_req_diff, params, vis)
            optimized_floor[scene][im] = {"R": X[0].rotation.tolist(), "t": (X[0].translation).tolist()}
    
        with open("/local2/homes/malenma3/collision-pose/eval/data/" + "ycbv_bop_floor_poses_1mm_res_optimized.json", "w") as f:
            json.dump(optimized_floor, f)

def load_static(floor_poses_name:Path):
    mesh_loader = hppfcl.MeshLoader()
    mesh = mesh_loader.load(str(FLOOR_MESH_PATH), np.array(3*[0.01]))
    mesh.buildConvexHull(True, "Qt")
    floor_mesh = mesh.convex
    with open(FLOOR_POSES_PATH / floor_poses_name, "r") as f:
        floor_se3s = json.load(f)
    return floor_mesh, floor_se3s

def save_optimized_bop(input_csv_name:str, output_csv_name:str,
                       dataset_name: str, use_floor:Union[None, str],
                       params:dict = None, vis:bool = False):
    rigid_objects = load_meshes(MESHES_PATH / dataset_name)
    rigid_objects_decomp = load_multi_convex_meshes(MESHES_DECOMP_PATH / dataset_name)
    if vis:
        rigid_objects_vis = load_meshes(MESHES_PATH / dataset_name, convex=False)
    scenes = load_csv(POSES_OUTPUT_PATH / dataset_name / input_csv_name)
    if use_floor != None:
        floor_mesh, floor_se3s = load_static(use_floor)
    args = SelectStrategyConfig(1e-2, 100)
    col_req, col_req_diff = select_strategy(args)

    with open(POSES_OUTPUT_PATH / dataset_name / output_csv_name, "w") as f:
        f.write("scene_id,im_id,obj_id,score,R,t,time\n")

    for scene in tqdm(scenes):
        for im in tqdm(scenes[scene]):
            curr_labels = []
            curr_meshes = []
            curr_meshes_decomp = []
            if vis:
                curr_meshes_vis = []
            else:
                curr_meshes_vis = None
            wMo_lst = []
            # Load info about each object in the scene
            for label, R_o, t_o in zip(scenes[scene][im]["obj_id"], scenes[scene][im]["R"], scenes[scene][im]["t"]):
                R_o = np.array(R_o).reshape(3, 3)
                t_o = np.array(t_o)
                wMo = pin.SE3(R_o, t_o)
                curr_labels.append(label)
                curr_meshes.append(rigid_objects[label])
                if vis:
                    curr_meshes_vis.append(rigid_objects_vis[label])
                curr_meshes_decomp.append(rigid_objects_decomp[label])
                wMo_lst.append(wMo)
            if use_floor:
                wMs = floor_se3s[str(scene)][str(im)]
                wMs, stat_meshes = ([], []) if wMs is None else ([pin.SE3(np.array(wMs["R"]), np.array(wMs["t"]))], [floor_mesh])
                dc_scene = DiffColScene(curr_meshes, stat_meshes, wMs, curr_meshes_decomp, pre_loaded_meshes=True)
            else:
                dc_scene = DiffColScene(curr_meshes, [], [], curr_meshes_decomp, pre_loaded_meshes=True)
            start_time = time.time()
            X = optim(dc_scene, wMo_lst, col_req, col_req_diff, params, curr_meshes_vis)
            optim_time = (time.time() - start_time)
            for i in range(len(X)):
                # One CSV row
                R = " ".join(str(item) for item in X[i].rotation.reshape(9).tolist())
                t = " ".join(str(item) for item in (X[i].translation*1000).tolist())
                csv_line = [scene, im, curr_labels[i], 1.0, R, t, optim_time]
                with open(POSES_OUTPUT_PATH / dataset_name / output_csv_name, "a") as f:
                    f.write(",".join([str(x) for x in csv_line]) + "\n")

if __name__ == "__main__":
    params = {
        "N_step": 1000,
        "coll_grad_scale": 0.01,
        "learning_rate": 0.001,
        "step_lr_decay": 0.5,
        "step_lr_freq": 100,
        "Q": [0.01, 0.06, 0.26],
        "method": "GD",
        "method_params": None
    }
    params_try = [  {"N_step": 1000,"coll_grad_scale": 0.2,"learning_rate": 0.0001,"step_lr_decay": 1,"step_lr_freq": 1000, 
                    "Q": [0.01, 0.24, 0.26],"method": "GD","method_params": None},
                    {"N_step": 1000,"coll_grad_scale": 0.2,"learning_rate": 0.0001,"step_lr_decay": 1,"step_lr_freq": 1000,
                    "Q": [0.005, 0.12, 0.26],"method": "GD","method_params": None},
                    {"N_step": 1000,"coll_grad_scale": 0.2,"learning_rate": 0.0001,"step_lr_decay": 1,"step_lr_freq": 1000,
                    "Q": [0.0025, 0.06, 0.26],"method": "GD","method_params": None},
                    {"N_step": 1000,"coll_grad_scale": 0.2,"learning_rate": 0.0001,"step_lr_decay": 1,"step_lr_freq": 1000,
                    "Q": [0.01, 0.06, 0.26],"method": "GD","method_params": None},
                    ]
    if len(sys.argv) > 1:
        param_num = int(sys.argv[1])
    else:
        param_num = int(input("Select param num: "))
    input_csv_name = "gt_refiner_final_ycbv-test.csv" # INPUT
    params = params_try[param_num]
    dataset_name = "ycbv" # INPUT
    output_csv_name = Path("GD_WO_SCHED_WITH_STATIC_LARGER_Q_LR_00001_CGS_02") #INPUT
    output_csv_name = output_csv_name / (f"{params['Q'][0]}-{params['Q'][1]}-{params['Q'][2]}_{dataset_name}-test".replace(".","") + ".csv") #(f"{params['coll_grad_scale']}-{params['learning_rate']}-{params['step_lr_decay']}-{params['step_lr_freq']}_{dataset_name}-test".replace(".","") + ".csv") #INPUT
    print(f"File name: {output_csv_name}")
    use_floor = "ycbv_bop_floor_poses_1mm_res_optimized.json" #INPUT str of None
    vis = False #INPUT
    save_optimized_bop(input_csv_name, output_csv_name, dataset_name, use_floor, params, vis)