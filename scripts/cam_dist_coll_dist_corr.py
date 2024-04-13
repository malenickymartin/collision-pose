import json, csv
import pinocchio as pin
import hppfcl
import numpy as np
from matplotlib import pyplot as plt

from eval.eval_utils import get_dist_decomp, load_csv, load_multi_convex_meshes, load_meshes
from config import POSES_OUTPUT_PATH, FLOOR_MESH_PATH, MESHES_DECOMP_PATH, MESHES_PATH, FLOOR_POSES_PATH

ds_name = "tless_one" # <====== INPUT
csv_file_names = [f"refiner-final_{ds_name}-test.csv",
                  f"2-001-075-50-01-0245-051-gt_{ds_name}-test.csv",
                  f"gravity-2-001-075-50-01-0245-051-gt_{ds_name}-test.csv"] # <====== INPUT
csv_vis_names = ["Megapose", "Collisions", "Collisions+Gravity"] # <====== INPUT
floor_poses_name = "tless_one_synt_floor_gt.json"  # <====== INPUT

meshes_ds_name = ""
if ds_name[:4] == "ycbv":
    meshes_ds_name = "ycbv"
elif ds_name[:5] == "tless":
    meshes_ds_name = "tless"
else:
    meshes_ds_name = ds_name


path_convex_meshes = MESHES_DECOMP_PATH / meshes_ds_name
path_meshes = MESHES_PATH / meshes_ds_name

floor_poses_path = FLOOR_POSES_PATH / floor_poses_name

gt_poses = load_csv(POSES_OUTPUT_PATH / ds_name / f"gt_{ds_name}-test.csv")
pred_poses = {}
for csv_name in zip(csv_vis_names, csv_file_names):
    pred_poses[csv_name[0]] = load_csv(POSES_OUTPUT_PATH / ds_name / csv_name[1])

loader = hppfcl.MeshLoader()
floor_path = str(FLOOR_MESH_PATH)
floor_hppfcl: hppfcl.BVHModelBase = loader.load(floor_path, scale = np.array([0.0005]*3))
floor_hppfcl.buildConvexHull(True, "Qt")
floor_mesh = floor_hppfcl.convex

with open(floor_poses_path, "r") as f:
    floor_poses = json.load(f)   

rigid_objects_decomp = load_multi_convex_meshes(path_convex_meshes)
rigid_objects = load_meshes(path_meshes)


all_err_points = {}
all_dist_points = {}

all_per_obj_err_points = {}
all_per_obj_dist_points = {}

for pred in pred_poses:

    err_points = []
    dist_points = []

    per_obj_err_points = {}
    per_obj_dist_points = {}

    for scene in gt_poses:
        for im in gt_poses[scene]:
            for obj in range(len(gt_poses[scene][im]["obj_id"])):
                assert gt_poses[scene][im]["obj_id"][obj] == pred_poses[pred][scene][im]["obj_id"][obj]
                gt_pose = pin.SE3(gt_poses[scene][im]["R"][obj], gt_poses[scene][im]["t"][obj])
                pred_pose = pin.SE3(pred_poses[pred][scene][im]["R"][obj], pred_poses[pred][scene][im]["t"][obj])
                floor_pose = pin.SE3(np.array(floor_poses[str(scene)][str(im)]["R"]), np.array(floor_poses[str(scene)][str(im)]["t"]))
                
                floor_dist = get_dist_decomp([floor_mesh], rigid_objects_decomp[gt_poses[scene][im]["obj_id"][obj]], floor_pose, pred_pose)
                cam_dist = np.linalg.norm(gt_pose.translation)

                t_error = np.linalg.norm(gt_pose.translation - pred_pose.translation)

                if gt_poses[scene][im]["obj_id"][obj] not in per_obj_err_points:
                    per_obj_err_points[gt_poses[scene][im]["obj_id"][obj]] = []
                    per_obj_dist_points[gt_poses[scene][im]["obj_id"][obj]] = []
                per_obj_err_points[gt_poses[scene][im]["obj_id"][obj]].append((cam_dist, t_error))
                per_obj_dist_points[gt_poses[scene][im]["obj_id"][obj]].append((cam_dist, floor_dist))

                dist_points.append((cam_dist, floor_dist))
                err_points.append((cam_dist, t_error))
    
    all_err_points[pred] = err_points
    all_dist_points[pred] = dist_points

    all_per_obj_err_points[pred] = per_obj_err_points
    all_per_obj_dist_points[pred] = per_obj_dist_points

used_colors = ["b", "r", "y", "c", "m", "g", "k"]

for i, err_points_label in enumerate(all_err_points):
    err_points = np.array(all_err_points[err_points_label])
    if i == 0:
        err_points += 0.001
    plt.scatter(err_points[:, 0], err_points[:, 1], label=err_points_label, color=used_colors[i], alpha=0.5)
plt.grid()
plt.title("Correlation between camera distance and translation error on synthetic dataset containing one object")
plt.xlabel("Camera distance [m]")
plt.ylabel("Translation error [m]")
plt.legend()
plt.show()

for i, dist_points_label in enumerate(all_dist_points):
    dist_points = np.array(all_dist_points[dist_points_label])
    if i == 0:
        dist_points += 0.001
    plt.scatter(dist_points[:, 0], dist_points[:, 1], label=dist_points_label, color=used_colors[i], alpha=0.5)
plt.grid()
plt.title("Correlation between camera distance and collision distance on synthetic dataset containing one object")
plt.xlabel("Camera distance [m]")
plt.ylabel("Collision distance [m]")
plt.legend()
plt.show()

assert all_per_obj_err_points[csv_vis_names[0]].keys() == all_per_obj_err_points[csv_vis_names[1]].keys()
assert all_per_obj_err_points[csv_vis_names[0]].keys() == all_per_obj_err_points[csv_vis_names[2]].keys()
assert all_per_obj_dist_points[csv_vis_names[0]].keys() == all_per_obj_dist_points[csv_vis_names[1]].keys()
assert all_per_obj_dist_points[csv_vis_names[0]].keys() == all_per_obj_dist_points[csv_vis_names[2]].keys()

obj_names = list(all_per_obj_err_points[csv_vis_names[0]].keys())

rows = int(np.floor(np.sqrt(len(obj_names))))
cols = int(np.ceil(len(obj_names) / rows))

for i, obj_id in enumerate(obj_names):
    plt.subplot2grid((rows, cols), (i // cols, i % cols))
    for j, csv_name in enumerate(csv_vis_names):
        err_points = np.array(all_per_obj_err_points[csv_name][obj_id])
        if j == 0:
            err_points += 0.001
        plt.scatter(err_points[:, 0], err_points[:, 1], label=csv_name, color=used_colors[j], alpha=0.5)
    plt.grid()
    plt.title(f"Object {obj_id}")
    plt.xlabel("Camera distance [m]")
    plt.ylabel("Translation error [m]")
    plt.legend()
plt.suptitle("Correlation between camera distance and translation error on synthetic dataset containing one object")
plt.show()


for i, obj_id in enumerate(obj_names):
    plt.subplot2grid((rows, cols), (i // cols, i % cols))
    for j, csv_name in enumerate(csv_vis_names):
        dist_points = np.array(all_per_obj_dist_points[csv_name][obj_id])
        if j == 0:
            dist_points += 0.001
        plt.scatter(dist_points[:, 0], dist_points[:, 1], label=csv_name, color=used_colors[j], alpha=0.5)
    plt.grid()
    plt.title(f"Object {obj_id}")
    plt.xlabel("Camera distance [m]")
    plt.ylabel("Collision distance [m]")
    plt.legend()
plt.suptitle("Correlation between camera distance and collision distance on synthetic dataset containing one object")
plt.show()
