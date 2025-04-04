import time
from tqdm import tqdm
import numpy as np
import pinocchio as pin
import json
from typing import Union
from argparse import ArgumentParser

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

from eval.eval_utils import load_meshes, load_meshes_decomp, load_csv, load_mesh, load_static

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
    print("Loading meshes")
    rigid_objects = load_meshes(MESHES_PATH / dataset_name)
    rigid_objects_decomp = load_meshes_decomp(MESHES_DECOMP_PATH / dataset_name)
    if vis:
        rigid_objects_vis = load_meshes(MESHES_PATH / dataset_name, convex=False)
    print("Loading input CSV")
    scenes = load_csv(POSES_OUTPUT_PATH / dataset_name / input_csv_name)
    print("Loading floor")
    if use_floor != None:
        floor_mesh, floor_se3s = load_static(use_floor)
    if vis and use_floor != None:
        floor_mesh_vis = [load_mesh(FLOOR_MESH_PATH, convex=False)]
    else:
        floor_mesh_vis = []
    pydiffcol.set_seed(0)
    col_req, col_req_diff = select_strategy(SelectStrategyConfig())

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
    parser = ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, required=True,
                        help="Name of the dataset")
    parser.add_argument("--init_poses", "-i", type=str, required=True,
                        help="Name of the input poses .csv file relative to the POSES_OUTPUT_PATH/dataset directory.")
    parser.add_argument("--floor", "-f", default=None,
                        help="Name of the floor .json file relative to the FLOOR_POSES_PATH directory.")
    parser.add_argument('--version', '-v', type=str, default="",
                        help="Version name which will be added to the output file name.")
    parser.add_argument("--vis", action="store_true",
                        help="Visualize the init and optimized poses and plot the costs and gradients for each image.")

    params = parser.add_argument_group("Optimization parameters")
    params.add_argument("--N_step", type=int, default=3000)
    params.add_argument("--g_grad_scale", type=float, default=1)
    params.add_argument("--coll_grad_scale", type=float, default=1)
    params.add_argument("--learning_rate", type=float, default=0.0001)
    params.add_argument("--step_lr_decay", type=float, default=1)
    params.add_argument("--step_lr_freq", type=int, default=1000)
    params.add_argument("--std_xy", type=float, default=0.05)
    params.add_argument("--std_z", type=float, default=0.49)
    params.add_argument("--std_theta", type=float, default=0.26)

    args = parser.parse_args()

    assert (POSES_OUTPUT_PATH / args.dataset / args.init_poses).is_file(), f"Input poses file does not exist: {POSES_OUTPUT_PATH / args.dataset / args.init_poses}"
    assert (FLOOR_POSES_PATH / args.floor).is_file() if args.floor else True, f"Floor poses file does not exist: {FLOOR_POSES_PATH / args.floor}"

    params = {
        "N_step": args.N_step,
        "g_grad_scale": args.g_grad_scale,
        "coll_grad_scale": args.coll_grad_scale,
        "learning_rate": args.learning_rate,
        "step_lr_decay": args.step_lr_decay,
        "step_lr_freq": args.step_lr_freq,
        "std_xy_z_theta": [args.std_xy, args.std_z, args.std_theta]
    }
    
    if args.floor == None and params["g_grad_scale"] != 0:
        print("No floor specified, but gravity gradient scale is not 0. Please specify a floor or set gravity to zero.")
        exit()

    floor_file_name = f"true_" if args.floor != None else "false_"
    output_csv_name = (f"{args.version}"
                       f"{params['N_step']}-"
                       f"{params['g_grad_scale']}-"
                       f"{params['coll_grad_scale']}-"
                       f"{params['learning_rate']}-"
                       f"{params['step_lr_decay']}-"
                       f"{params['step_lr_freq']}-"
                       f"{params['std_xy_z_theta'][0]}-{params['std_xy_z_theta'][1]}-{params['std_xy_z_theta'][2]}-"
                       f"{floor_file_name}"
                       f"{args.dataset}-test").replace(".","") + ".csv"
    (POSES_OUTPUT_PATH / args.dataset / output_csv_name).parent.mkdir(parents=True, exist_ok=True)
    print(f"Output file name: {output_csv_name}")

    save_optimized_bop(args.init_poses, output_csv_name, args.dataset, args.floor, params, args.vis)