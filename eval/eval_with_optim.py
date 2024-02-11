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
from pydiffcol.utils_render import create_visualizer

from scene import DiffColScene, draw_scene, SelectStrategyConfig
from optim import (
    perception_res_grad,
    update_est
)
from eval.eval_utils import get_se3_from_mp_json, get_se3_from_bp_cam


def optim(dc_scene:DiffColScene, wMo_lst_init, col_req, col_req_diff, N_step:int = 100, visualize:bool = False):
    X = deepcopy(wMo_lst_init)
    X_lst = []

    N_SHAPES = len(dc_scene.shapes)

    # consts
    learning_rate = 0.0001
    beta1_adam = 0.9
    beta2_adam = 0.999
    eps_adam = 1e-8
    scale_grad_col = 1

    m_adam = np.zeros(6*N_SHAPES)
    v_adam = np.zeros(6*N_SHAPES)
    dx = np.zeros(6*N_SHAPES)
    
    for i in range(N_step):
        X_lst.append(deepcopy(X))
        if i % 100 == 0: 
            learning_rate /= 2

        X_eval = X

        _, grad_c_obj = dc_scene.compute_diffcol(X_eval, col_req, col_req_diff)
        _, grad_c_stat = dc_scene.compute_diffcol_static(X_eval, col_req, col_req_diff)
        grad_c = grad_c_obj + grad_c_stat
        # cost_c, grad_c = 0.0, np.zeros(dx.shape)
        _, grad_p = perception_res_grad(X_eval, wMo_lst_init)
        # res_p, grad_p = np.zeros(dx.shape), np.zeros(dx.shape)
        grad = scale_grad_col*grad_c + grad_p

        m_adam = beta1_adam*m_adam + (1-beta1_adam)*grad
        v_adam = beta2_adam*v_adam + (1-beta2_adam)*(grad**2)
        dx = - learning_rate * m_adam / (np.sqrt(v_adam) + eps_adam)

        # state update
        X = update_est(X, dx)

    if visualize:
        print('Create vis')
        vis = create_visualizer()
        dc_scene.compute_diffcol(wMo_lst_init, col_req, col_req_diff, diffcol=False)
        dc_scene.compute_diffcol_static(wMo_lst_init, col_req, col_req_diff, diffcol=False)
        input("Continue?")
        print('Init!')
        draw_scene(vis, dc_scene.shapes, dc_scene.stat_shapes, wMo_lst_init, dc_scene.wMs_lst, dc_scene.col_res_pairs, dc_scene.col_res_pairs_stat, render_faces=False)
        time.sleep(5)
        print('Optimized!')
        draw_scene(vis, dc_scene.shapes, dc_scene.stat_shapes, X, dc_scene.wMs_lst, dc_scene.col_res_pairs, dc_scene.col_res_pairs_stat, render_faces=False)
        time.sleep(5)

        # Process
        print("Animation start!")
        for Xtmp in X_lst:
            # Recompute collision between pairs for visualization (TODO: should be stored)
            time.sleep(0.1)
            dc_scene.compute_diffcol(Xtmp, col_req, col_req_diff, diffcol=False)
            dc_scene.compute_diffcol_static(Xtmp, col_req, col_req_diff, diffcol=False)
            draw_scene(vis, dc_scene.shapes, dc_scene.stat_shapes, Xtmp, dc_scene.wMs_lst, dc_scene.col_res_pairs, dc_scene.col_res_pairs_stat, render_faces=False)
        print("Animation done!")
    
    return X

def create_mesh(mesh_loader, obj_path: str):
    mesh: hppfcl.BVHModelBase = mesh_loader.load(obj_path, np.array(3*[0.001]))
    mesh.buildConvexHull(True, "Qt")
    return mesh.convex

def main():

    visualize = True
    N_step = 1000

    args = SelectStrategyConfig()
    args.noise = 1e-2
    args.num_samples = 100
    col_req, col_req_diff = select_strategy(args)

    path_stat_objs = ["eval/data/floor.ply"]
    wMs_lst = [pin.SE3(np.eye(4))]

    path_objs_all = Path("eval/data/ycbv_convex_two/meshes")
    mesh_objs_dict = {}
    mesh_loader = hppfcl.MeshLoader()
    for mesh_dir_path in path_objs_all.iterdir():
        mesh_label = int(mesh_dir_path.name)
        mesh_path = mesh_dir_path / f"obj_{mesh_label:06d}.ply"
        mesh_objs_dict[mesh_label] = create_mesh(mesh_loader, str(mesh_path))
    
    path_wMo_all = Path("eval/data/ycbv_convex_two/happypose/outputs")
    gt_cam_path = Path("eval/data/ycbv_convex_two/train_pbr/000000/scene_camera.json")
    wMo_lst_all = []
    label_objs_all = []
    scene_idxs = []
    cams = []
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
            wMo = se3_cam.inverse() * get_se3_from_mp_json(wMo_json[i])
            wMo_lst.append(wMo)
            label_objs.append(label)
        wMo_lst_all.append(wMo_lst)
        label_objs_all.append(label_objs)
        scene_idxs.append(scene_idx)
        cams.append(se3_cam)

    save_dir_path = Path("eval/data/ycbv_convex/happypose/outputs_coll")
    for i in tqdm(range(len(wMo_lst_all))):
        curr_meshes = []
        for l in label_objs_all[i]:
            curr_meshes.append(mesh_objs_dict[l])
        wMo_lst = wMo_lst_all[i]
        dc_scene = DiffColScene(curr_meshes, path_stat_objs, wMs_lst)
        X = optim(dc_scene, wMo_lst, col_req, col_req_diff, N_step, visualize)
        to_json = []
        for j in range(len(X)):
            xyzquat = pin.SE3ToXYZQUAT(cams[i] * X[j])
            json_dict = {"label":str(label_objs_all[i][j]), "TWO":[list(xyzquat[3:]), list(xyzquat[:3])]}
            to_json.append(json_dict)
        save_data_path = save_dir_path / f"object_data_{scene_idxs[i]}.json"
        save_data_path.write_text(json.dumps(to_json))


if __name__ == "__main__":
    main()

