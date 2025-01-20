import time
from tqdm import tqdm
import numpy as np
import pinocchio as pin
import json
from copy import deepcopy
import hppfcl
from robomeshcat import Scene, Object

from config import (MESHES_PATH,
                    FLOOR_MESH_PATH,
                    POSES_OUTPUT_PATH,
                    FLOOR_POSES_PATH)

from eval.eval_utils import load_csv

def load_static(floor_poses_name:str):
    """Loads floor mesh and poses from JSON file."""
    mesh_loader = hppfcl.MeshLoader()
    mesh = mesh_loader.load(str(FLOOR_MESH_PATH), np.array(3*[0.01]))
    mesh.buildConvexHull(True, "Qt")
    floor_mesh = mesh.convex
    with open(FLOOR_POSES_PATH / floor_poses_name, "r") as f:
        floor_se3s = json.load(f)
    return floor_mesh, floor_se3s

def visualize_scenes(dataset_name: str, floor_name: str, megapose_csv_name: str, optimized_csv_name: str):
    
    scenes_optim = load_csv(POSES_OUTPUT_PATH / dataset_name / optimized_csv_name)
    scenes_gt = load_csv(POSES_OUTPUT_PATH / dataset_name / f"gt_{dataset_name}-test.csv")
    scenes_mega = load_csv(POSES_OUTPUT_PATH / dataset_name / megapose_csv_name)
    _, floor_se3s = load_static(floor_name)

    meshes_ds_name = ""
    if dataset_name[:4] == "ycbv":
        meshes_ds_name = "ycbv"
    elif dataset_name[:5] == "tless":
        meshes_ds_name = "tless"
    else:
        meshes_ds_name = dataset_name

    maximmum_number_of_objects_of_same_label_in_one_scene_in_dataset = 1
    if dataset_name == "tless":
        maximmum_number_of_objects_of_same_label_in_one_scene_in_dataset = 7

    scene_rmc = Scene()
    floor_rmc = Object.create_mesh(
        FLOOR_MESH_PATH.parent / "floor" / "floor.obj",
        scale=0.001,
        texture=FLOOR_MESH_PATH.parent / "floor" / "Wood001_2K-JPG_Color.jpg",
        color=[0.5] * 3,
        )
    scene_rmc.add_object(floor_rmc)
    objects_all_rmc = {}
    print("Loading RMC meshes")
    for mesh_dir_path in tqdm((MESHES_PATH / meshes_ds_name).iterdir()):
        l = int(mesh_dir_path.name)
        mesh_path = mesh_dir_path /f"obj_{l:06d}.ply"
        texture_path = mesh_dir_path /f"obj_{l:06d}.png"
        objects_all_rmc[str(l)] = []
        for i in range(maximmum_number_of_objects_of_same_label_in_one_scene_in_dataset):
            o = Object.create_mesh(
                mesh_path,
                scale=0.001,
                #texture=texture_path,
                color=[0.8] * 3,
                )
            objects_all_rmc[str(l)].append(o)

    while True:
        scene = np.random.choice(list(scenes_gt.keys()))
        im = np.random.choice(list(scenes_gt[scene].keys()))
        scene = int(input("Scene: "))
        im = int(input("Image: "))
        #print(f"Scene: {scene}, Image: {im}")

        wMs = floor_se3s[str(scene)][str(im)]
        wMs = None if wMs is None else pin.SE3(np.array(wMs["R"]), np.array(wMs["t"]))
        if wMs == None:
            if floor_rmc in list(scene_rmc.objects.values()):
                scene_rmc.remove_object(floor_rmc)
        else:
            if not floor_rmc in list(scene_rmc.objects.values()):
                scene_rmc.add_object(floor_rmc)
            floor_rmc.pose = wMs.homogeneous

        wMo_gt = {}
        labels_gt = []
        for label, R_o, t_o in zip(scenes_gt[scene][im]["obj_id"], scenes_gt[scene][im]["R"], scenes_gt[scene][im]["t"]):
            R_o = np.array(R_o).reshape(3, 3)
            t_o = np.array(t_o)
            wMo = pin.SE3(R_o, t_o)
            if label not in wMo_gt:
                wMo_gt[label] = []
            wMo_gt[label].append(wMo)
            labels_gt.append(label)

        wMo_optim = {}
        labels_optim = []
        for label, R_o, t_o in zip(scenes_optim[scene][im]["obj_id"], scenes_optim[scene][im]["R"], scenes_optim[scene][im]["t"]):
            R_o = np.array(R_o).reshape(3, 3)
            t_o = np.array(t_o)
            wMo = pin.SE3(R_o, t_o)
            if label not in wMo_optim:
                wMo_optim[label] = []
            wMo_optim[label].append(wMo)
            labels_optim.append(label)

        wMo_mega = {}
        labels_mega = []
        for label, R_o, t_o in zip(scenes_mega[scene][im]["obj_id"], scenes_mega[scene][im]["R"], scenes_mega[scene][im]["t"]):
            R_o = np.array(R_o).reshape(3, 3)
            t_o = np.array(t_o)
            wMo = pin.SE3(R_o, t_o)
            if label not in wMo_mega:
                wMo_mega[label] = []
            wMo_mega[label].append(wMo)
            labels_mega.append(label)

        for label in labels_gt:
            assert labels_optim.count(label) == labels_gt.count(label)
            assert labels_mega.count(label) == labels_gt.count(label)
        for label in labels_optim:
            assert labels_gt.count(label) == labels_optim.count(label)
            assert labels_mega.count(label) == labels_optim.count(label)
        for label in labels_mega:
            assert labels_gt.count(label) == labels_mega.count(label)
            assert labels_optim.count(label) == labels_mega.count(label)

        curr_objects_rmc = {}
        for label in labels_gt:
            if label not in curr_objects_rmc:
                curr_objects_rmc[label] = []
            curr_objects_rmc[label].append(objects_all_rmc[label][len(curr_objects_rmc[label])])
            
        for label in curr_objects_rmc:
            for curr_object_rmc in curr_objects_rmc[label]:
                scene_rmc.add_object(curr_object_rmc)

        vis_configuration = "i"
        while True:
            if vis_configuration == "i":
                for label in wMo_mega:
                    for i, wMo in enumerate(wMo_mega[label]):
                        curr_objects_rmc[label][i].pose = wMo.homogeneous
            elif vis_configuration == "o":
                for label in wMo_optim:
                    for i, wMo in enumerate(wMo_optim[label]):
                        curr_objects_rmc[label][i].pose = wMo.homogeneous
            elif vis_configuration == "g":
                for label in wMo_gt:
                    for i, wMo in enumerate(wMo_gt[label]):
                        curr_objects_rmc[label][i].pose = wMo.homogeneous
            elif vis_configuration == "q":
                break
            vis_configuration = input("Visualize configuration (i/o/g/q): ")
            time.sleep(3)

        for object_rmc in list(scene_rmc.objects.values()):
            if object_rmc != floor_rmc:
                scene_rmc.remove_object(object_rmc)

if __name__ == "__main__":
    ds_name = "ycbv"
    floor_file_names = {"hopevideo":"hope_bop_floor_poses_1mm_res_optimized.json",
                    "tless":"tless_bop_floor_poses_1mm_res_optimized.json",
                    "ycbv":"ycbv_bop_floor_poses_1mm_res_optimized.json",
                    "ycbvone":"ycbv_one_synt_floor_gt.json",
                    "tlessone":"tless_one_synt_floor_gt.json"}
    mega_csv_names = {"hopevideo":"refiner-final-filtered_hopevideo-test.csv",
                       "tless":"refiner-final-filtered_tless-test.csv",
                       "ycbv":"refiner-final-filtered_ycbv-test.csv",
                       "ycbvone":"refiner-final_ycbvone-test.csv",
                       "tlessone":"refiner-final_tlessone-test.csv"}
    optim_csv_name = f"final-fog-exp-test/1-1-00001-1-1000-005-049-026-0-optimized_{ds_name}-test.csv"
    visualize_scenes(ds_name, floor_file_names[ds_name], mega_csv_names[ds_name], optim_csv_name)