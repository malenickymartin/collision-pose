from scene import DiffColScene, draw_scene, SelectStrategyConfig
from eval.eval_utils import load_csv, load_meshes, load_multi_convex_meshes, load_mesh
from config import POSES_OUTPUT_PATH, FLOOR_POSES_PATH, MESHES_PATH, MESHES_DECOMP_PATH, FLOOR_MESH_PATH
from pydiffcol.utils_render import create_visualizer
from pydiffcol.utils import select_strategy

import numpy as np
import json
import pinocchio as pin
import hppfcl

dataset_name = 'ycbv'
input_csv_name = 'gt_ycbv-test.csv'
floor_name = 'ycbv_bop_floor_poses_1mm_res_optimized.json'

def load_static(floor_poses_name:str):
    mesh_loader = hppfcl.MeshLoader()
    mesh = mesh_loader.load(str(FLOOR_MESH_PATH), np.array(3*[0.01]))
    mesh.buildConvexHull(True, "Qt")
    floor_mesh = mesh.convex
    with open(FLOOR_POSES_PATH / floor_poses_name, "r") as f:
        floor_se3s = json.load(f)
    return floor_mesh, floor_se3s

scenes = load_csv(POSES_OUTPUT_PATH / dataset_name / input_csv_name)
floor_mesh, floor_se3s = load_static(floor_name)
rigid_objects = load_meshes(MESHES_PATH / dataset_name)
rigid_objects_decomp = load_multi_convex_meshes(MESHES_DECOMP_PATH / dataset_name)
rigid_objects_vis = load_meshes(MESHES_PATH / dataset_name, convex=False)
floor_mesh_vis = [load_mesh(FLOOR_MESH_PATH, convex=False)]

args = SelectStrategyConfig(1e-2, 100, 1, "finite_differences")
col_req, col_req_diff = select_strategy(args)

for scene in scenes:
    vis = create_visualizer(grid=True, axes=True)
    for im in scenes[scene]:
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
            curr_meshes_vis.append(rigid_objects_vis[label])
            curr_meshes_decomp.append(rigid_objects_decomp[label])
            wMo_lst.append(wMo)
        wMs = floor_se3s[str(scene)][str(im)]
        wMs, stat_meshes = ([], []) if wMs is None else ([pin.SE3(np.array(wMs["R"]), np.array(wMs["t"]))], [floor_mesh])
        dc_scene = DiffColScene(curr_meshes, stat_meshes, wMs, curr_meshes_decomp, pre_loaded_meshes=True)
        dc_scene.compute_diffcol(wMo_lst, col_req, col_req_diff, diffcol=False)
        dc_scene.compute_diffcol_static(wMo_lst, col_req, col_req_diff, diffcol=False)
        draw_scene(vis, curr_meshes_vis, floor_mesh_vis, wMo_lst, dc_scene.wMs_lst, dc_scene.col_res_pairs, dc_scene.col_res_pairs_stat)
        input(f"Scene {scene}, image {im} - Press Enter to continue...")
    
"""
List of bad floor poses:
Scene 4, im 448
Scene 4, im 455
Scene 4, im 462
Scene 4, im 469
Scene 4, im 479

Scene 13, im 450
Scene 13, im 462
Scene 13, im 478
Scene 13, im 486
Scene 13, im 500

Scene 9, im 437
Scene 9, im 451
Scene 9, im 465
Scene 9, im 477
Scene 9, im 484
Scene 9, im 490
Scene 9, im 495

Scene 5, im 450
Scene 5, im 458
Scene 5, im 463
Scene 5, im 468
Scene 5, im 475
Scene 5, im 493

Scene 2, im 434
Scene 2, im 439
Scene 2, im 446
Scene 2, im 455
Scene 2, im 459
Scene 2, im 463
Scene 2, im 470
Scene 2, im 478
Scene 2, im 489
Scene 2, im 494

Scene 8, im 459
Scene 8, im 465
Scene 8, im 471
Scene 8, im 479
Scene 8, im 484
Scene 8, im 495

Scene 10, im 441
Scene 10, im 450
Scene 10, im 457
Scene 10, im 465
Scene 10, im 475
Scene 10, im 482
Scene 10, im 496

Scene 12, im 438
Scene 12, im 455
Scene 12, im 471
Scene 12, im 482
Scene 12, im 490

Scene 1, im 434
Scene 1, im 442
Scene 1, im 455
Scene 1, im 464
Scene 1, im 474
Scene 1, im 488
Scene 1, im 493

Scene 3, im 446
Scene 3, im 454
Scene 3, im 459
Scene 3, im 466
Scene 3, im 476
Scene 3, im 493

Scene 6, im 434
Scene 6, im 442
Scene 6, im 462
Scene 6, im 469
Scene 6, im 475
Scene 6, im 485
Scene 6, im 496
Scene 6, im 501

Scene 7, im 435
Scene 7, im 447
Scene 7, im 450
Scene 7, im 458
Scene 7, im 468
Scene 7, im 478
Scene 7, im 486
Scene 7, im 493

Scene 18, im 439
Scene 18, im 452
Scene 18, im 461
Scene 18, im 466
Scene 18, im 473
Scene 18, im 480
Scene 18, im 487
Scene 18, im 491
Scene 18, im 496

Scene 11, im 441
Scene 11, im 452
Scene 11, im 459
Scene 11, im 469
Scene 11, im 476
Scene 11, im 486
Scene 11, im 497
"""

wrong_floor_poses = {"4": [448, 455, 462, 469, 479],
                     "13": [450, 462, 478, 486, 500],
                     "9": [437, 451, 465, 477, 484, 490, 495],
                     "5": [450, 458, 463, 468, 475, 493],
                     "2": [434, 439, 446, 455, 459, 463, 470, 478, 489, 494],
                     "8": [459, 465, 471, 479, 484, 495],
                     "10": [441, 450, 457, 465, 475, 482, 496],
                     "12": [438, 455, 471, 482, 490],
                     "1": [434, 442, 455, 464, 474, 488, 493],
                     "3": [446, 454, 459, 466, 476, 493],
                     "6": [434, 442, 462, 469, 475, 485, 496, 501],
                     "7": [435, 447, 450, 458, 468, 478, 486, 493],
                     "18": [439, 452, 461, 466, 473, 480, 487, 491, 496],
                     "11": [441, 452, 459, 469, 476, 486, 497]}