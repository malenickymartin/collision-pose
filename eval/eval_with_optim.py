import time
from copy import deepcopy
from tqdm import tqdm
import sys
import numpy as np
import pinocchio as pin
from pathlib import Path
import json
import hppfcl
from typing import Union

import pydiffcol
from pydiffcol.utils import (
    select_strategy
)

from src.scene import DiffColScene, SelectStrategyConfig
from src.optimizer import optim
from config import (MESHES_PATH,
                    MESHES_DECOMP_PATH,
                    FLOOR_MESH_PATH,
                    DATASETS_PATH,
                    POSES_OUTPUT_PATH,
                    FLOOR_POSES_PATH)

from eval.eval_utils import get_se3_from_mp_json, get_se3_from_bp_cam, load_meshes, load_meshes_decomp, load_csv, load_mesh

def save_optimized_floor(dataset_name:str, floor_name:str, params:dict = None, vis:bool = None):
    """Optimizes pose of floor in YCBV BOP dataset using collision detection.
    Inputs:
    - dataset_name: str, name of the dataset
    - floor_name: str, name of the floor
    - params: dict, optimization parameters
    - vis: bool, whether to visualize the optimization process
    Returns: None
    """

    rigid_objects = load_meshes(MESHES_PATH / dataset_name)
    rigid_objects_decomp = load_meshes_decomp(MESHES_DECOMP_PATH / dataset_name)
    floor_mesh, floor_se3s = load_static(floor_name)
    args = SelectStrategyConfig(1e-2, 100)
    col_req, col_req_diff = select_strategy(args)
    if vis:
        rigid_objects_vis = load_meshes(MESHES_PATH / dataset_name, convex=False)
        curr_meshes_vis = [load_mesh(FLOOR_MESH_PATH, convex=False)]
    else:
        curr_meshes_vis = []

    optimized_floor = {}

    for scene in tqdm(floor_se3s):
        with open(DATASETS_PATH / dataset_name / f"{int(scene):06d}" / "scene_gt.json", "r") as f:
            gt_poses_all = json.load(f)
        optimized_floor[scene] = {}
        for im in tqdm(floor_se3s[scene]):
            curr_stat_meshes = []
            curr_stat_meshes_decomp = []
            wMs_lst = []
            curr_meshes_stat_vis = []
            # Load info about each object in the scene
            for obj in gt_poses_all[im]:
                wMs_lst.append(pin.SE3(np.array(obj["cam_R_m2c"]).reshape(3,3), np.array(obj["cam_t_m2c"])/1000))
                curr_stat_meshes.append(rigid_objects[str(obj["obj_id"])])
                curr_stat_meshes_decomp.append(rigid_objects_decomp[str(obj["obj_id"])])
                if vis:
                    curr_meshes_stat_vis.append(rigid_objects_vis[str(obj["obj_id"])])
            wMo = floor_se3s[str(scene)][str(im)]
            wMo_lst, curr_meshes = (None, None) if wMo is None else ([pin.SE3(np.array(wMo["R"]), np.array(wMo["t"]))], [floor_mesh])
            if wMo_lst is None:
                optimized_floor[scene][im] = None
                continue
            dc_scene = DiffColScene(curr_meshes, curr_stat_meshes, wMs_lst, [], curr_stat_meshes_decomp, pre_loaded_meshes=True)
            X = optim(dc_scene, wMo_lst, col_req, col_req_diff, params, curr_meshes_vis, curr_meshes_stat_vis)
            optimized_floor[scene][im] = {"R": X[0].rotation.tolist(), "t": (X[0].translation).tolist()}
        with open(FLOOR_POSES_PATH / (floor_name[:-5] + "_optimized.json"), "w") as f:
            json.dump(optimized_floor, f)

def load_static(floor_poses_name:str):
    """Loads floor mesh and poses from JSON file."""
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
    """
    Optimizes poses of objects in BOP dataset using collision detection.
    Inputs:
    - input_csv_name: str, name of the input CSV file
    - output_csv_name: str, name of the output CSV file
    - dataset_name: str, name of the dataset
    - use_floor: str, name of the floor
    - params: dict, optimization parameters
    - vis: bool, whether to visualize the optimization process
    Returns: None
    """
    meshes_ds_name = ""
    if dataset_name[:4] == "ycbv":
        meshes_ds_name = "ycbv"
    elif dataset_name[:5] == "tless":
        meshes_ds_name = "tless"
    else:
        meshes_ds_name = dataset_name
    rigid_objects = load_meshes(MESHES_PATH / meshes_ds_name)
    rigid_objects_decomp = load_meshes_decomp(MESHES_DECOMP_PATH / meshes_ds_name)
    if vis:
        rigid_objects_vis = load_meshes(MESHES_PATH / meshes_ds_name, convex=False)
    scenes = load_csv(POSES_OUTPUT_PATH / dataset_name / input_csv_name)
    if use_floor != None:
        floor_mesh, floor_se3s = load_static(use_floor)
    if vis and use_floor != None:
        floor_mesh_vis = [load_mesh(FLOOR_MESH_PATH, convex=False)]
    else:
        floor_mesh_vis = []
    args = SelectStrategyConfig(params['noise'], params['gauss_samples'], params["max_neighbors_search_level"], "first_order_gaussian")
    pydiffcol.set_seed(0)
    col_req, col_req_diff = select_strategy(args)

    with open(POSES_OUTPUT_PATH / dataset_name / output_csv_name, "w") as f:
        f.write("scene_id,im_id,obj_id,score,R,t,time\n")
    for scene in tqdm(scenes):
        for im in tqdm(scenes[scene]):
            curr_labels = []
            curr_meshes = []
            curr_meshes_decomp = []
            curr_meshes_vis = []
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
            X = optim(dc_scene, wMo_lst, col_req, col_req_diff, params, curr_meshes_vis, floor_mesh_vis)
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
        "N_step": 3000,
        "g_grad_scale": 1,
        "coll_grad_scale": 1,
        "coll_exp_scale": 0,
        "learning_rate": 0.0001,
        "step_lr_decay": 1,
        "step_lr_freq": 1000,
        "std_xy_z_theta": [0.05, 0.49, 0.26],
        "noise": 1,
        "gauss_samples": 100,
        "max_neighbors_search_level": 1
    }
    
    floor_file_names = {"hopevideo":"hope_bop_floor_poses_1mm_res_optimized.json",
                        "tless":"tless_bop_floor_poses_1mm_res_optimized.json",
                        "ycbv":"ycbv_bop_floor_poses_1mm_res_optimized.json",
                        "ycbvone":"ycbv_one_synt_floor_gt.json",
                        "tlessone":"tless_one_synt_floor_gt.json"}
    floor_names = ["optimized", "none"]

    input_csv_names = {"hopevideo":"refiner-final-filtered_hopevideo-test.csv",
                       "tless":"refiner-final-filtered_tless-test.csv",
                       "ycbv":"gt-refiner-final_ycbv-test.csv",
                       "ycbvone":"refiner-final_ycbvone-test.csv",
                       "tlessone":"refiner-final_tlessone-test.csv"} # INPUT
    
    dataset_names = ["hopevideo","ycbv","tless","ycbvone","tlessone"] # INPUT
    vis = True #INPUT

    #dataset_name = dataset_names[int(sys.argv[1])]
    dataset_name = "ycbv"
    #params = params_try[int(sys.argv[2])]
    #floor_name = floor_names[int(sys.argv[3])]
    floor_name = floor_names[0]

    input_csv_name = input_csv_names[dataset_name]
    if floor_name == "none":
        use_floor = None
        if params["g_grad_scale"] != 0:
            exit()
    else:
        use_floor = floor_file_names[dataset_name]
    

    output_csv_name = ("TEST/"
                       f"{params['g_grad_scale']}-"
                       f"{params['coll_grad_scale']}-"
                       f"{params['learning_rate']}-"
                       f"{params['step_lr_decay']}-"
                       f"{params['step_lr_freq']}-"
                       f"{params['std_xy_z_theta'][0]}-{params['std_xy_z_theta'][1]}-{params['std_xy_z_theta'][2]}-"
                       f"{params['coll_exp_scale']}-"
                       f"{floor_name}_"
                       f"{dataset_name}-test").replace(".","") + ".csv"
    print(f"File name: {output_csv_name}")
    if floor_name == "none":
        use_floor = None
        if params["g_grad_scale"] != 0:
            exit()
    else:
        use_floor = floor_file_names[dataset_name]
    save_optimized_bop(input_csv_name, output_csv_name, dataset_name, use_floor, params, vis)