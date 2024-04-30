import json, csv
import pinocchio as pin
import hppfcl
import numpy as np
from matplotlib import pyplot as plt

from eval.eval_utils import get_dist_decomp, load_csv, load_meshes_decomp, load_meshes, get_dist_convex
from config import POSES_OUTPUT_PATH, FLOOR_MESH_PATH, MESHES_DECOMP_PATH, MESHES_PATH, FLOOR_POSES_PATH

all_ds_err_points = {}
all_ds_dist_points = {}
all_ds_dist_err_corr_points = {}

for ds_name in ["ycbvone", "tlessone"]: # <====== INPUT

    #ds_name = "tlessone" # <====== INPUT
    csv_file_names = [f"final-fog/0-2-00001-1-1000-005-049-026-0-optimized_{ds_name}-test.csv",
                    f"final-fog/2-2-00001-1-1000-005-049-026-0-optimized_{ds_name}-test.csv",
                    f"refiner-final_{ds_name}-test.csv",] # <====== INPUT
    csv_vis_names = ["Collisions", "Collisions+Gravity", "Megapose"] # <====== INPUT
    floor_poses_name = f"{ds_name[:-3]}_one_synt_floor_gt.json"  # <====== INPUT

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
    floor_hppfcl: hppfcl.BVHModelBase = loader.load(floor_path, scale = np.array([0.001]*3))
    floor_hppfcl.buildConvexHull(True, "Qt")
    floor_mesh = floor_hppfcl.convex

    with open(floor_poses_path, "r") as f:
        floor_poses = json.load(f)   

    rigid_objects_decomp = load_meshes_decomp(path_convex_meshes)
    rigid_objects = load_meshes(path_meshes)


    all_err_points = {}
    all_dist_points = {}
    all_dist_err_corr_points = {}

    all_per_obj_err_points = {}
    all_per_obj_dist_points = {}

    for pred in pred_poses:

        err_points = []
        dist_points = []
        dist_err_corr_points = []

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

                    if abs(floor_dist) > 0.4:
                        continue

                    if gt_poses[scene][im]["obj_id"][obj] not in per_obj_err_points:
                        per_obj_err_points[gt_poses[scene][im]["obj_id"][obj]] = []
                        per_obj_dist_points[gt_poses[scene][im]["obj_id"][obj]] = []
                    per_obj_err_points[gt_poses[scene][im]["obj_id"][obj]].append((cam_dist, t_error))
                    per_obj_dist_points[gt_poses[scene][im]["obj_id"][obj]].append((cam_dist, floor_dist))
                    dist_points.append((cam_dist, floor_dist))
                    err_points.append((cam_dist, t_error))
                    dist_err_corr_points.append((floor_dist, t_error))
        
        all_err_points[pred] = err_points
        all_dist_points[pred] = dist_points
        all_dist_err_corr_points[pred] = dist_err_corr_points

        all_per_obj_err_points[pred] = per_obj_err_points
        all_per_obj_dist_points[pred] = per_obj_dist_points
    
    all_ds_err_points[ds_name] = all_err_points
    all_ds_dist_points[ds_name] = all_dist_points
    all_ds_dist_err_corr_points[ds_name] = all_dist_err_corr_points


used_colors = ["r", "y", "b", "c", "m", "g", "k"]

############ BOTH DS ############

err_megapose_ycbv = np.array(all_ds_err_points["ycbvone"]["Megapose"])
err_coll_ycbv = np.array(all_ds_err_points["ycbvone"]["Collisions"])
err_coll_grav_ycbv = np.array(all_ds_err_points["ycbvone"]["Collisions+Gravity"])
dist_coll_ycbv = np.array(all_ds_dist_points["ycbvone"]["Collisions"])
dist_coll_grav_ycbv = np.array(all_ds_dist_points["ycbvone"]["Collisions+Gravity"])
dist_megapose_ycbv = np.array(all_ds_dist_points["ycbvone"]["Megapose"])
err_megapose_tless = np.array(all_ds_err_points["tlessone"]["Megapose"])
err_coll_tless = np.array(all_ds_err_points["tlessone"]["Collisions"])
err_coll_grav_tless = np.array(all_ds_err_points["tlessone"]["Collisions+Gravity"])
dist_coll_tless = np.array(all_ds_dist_points["tlessone"]["Collisions"])
dist_coll_grav_tless = np.array(all_ds_dist_points["tlessone"]["Collisions+Gravity"])
dist_megapose_tless = np.array(all_ds_dist_points["tlessone"]["Megapose"])

dist_err_corr_points_ycbv = np.array(all_ds_dist_err_corr_points["ycbvone"]["Collisions"])[::2]
dist_err_corr_points_grav_ycbv = np.array(all_ds_dist_err_corr_points["ycbvone"]["Collisions+Gravity"])[::2]
dist_err_corr_points_megapose_ycbv = np.array(all_ds_dist_err_corr_points["ycbvone"]["Megapose"])[::2]
dist_err_corr_points_tless = np.array(all_ds_dist_err_corr_points["tlessone"]["Collisions"])[::2]
dist_err_corr_points_grav_tless = np.array(all_ds_dist_err_corr_points["tlessone"]["Collisions+Gravity"])[::2]
dist_err_corr_points_megapose_tless = np.array(all_ds_dist_err_corr_points["tlessone"]["Megapose"])[::2]

# plt.scatter(dist_err_corr_points_megapose_ycbv[:, 0], dist_err_corr_points_megapose_ycbv[:, 1], label="Megapose - YCB-Video", color="tab:blue", marker="o", s=70)
# plt.scatter(dist_err_corr_points_megapose_tless[:, 0], dist_err_corr_points_megapose_tless[:, 1], label="Megapose - T-LESS", color="tab:brown", marker="o", s=70)
# plt.plot([-0.12, 0, 0.3], [0.12, 0, 0.3], color="black", linestyle="--", label="y=|x|")
# plt.grid()
# plt.title("Synthetic datasets containing one object", fontsize=20)
# plt.xlabel("Collision distance [m]", fontsize=20)
# plt.ylabel("Translation error [m]", fontsize=20)
# plt.legend(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xticks(fontsize=15)
# plt.show()

# plt.scatter(dist_err_corr_points_grav_ycbv[:, 0], dist_err_corr_points_grav_ycbv[:, 1], label="Our method (Floor collisions + Gravity) - YCB-Video", color="tab:red", marker="o", s=70)
# plt.scatter(dist_err_corr_points_grav_tless[:, 0], dist_err_corr_points_grav_tless[:, 1], label="Our method (Floor collisions + Gravity) - T-LESS", color="tab:olive", marker="o", s=70)
# plt.plot([-0.12, 0, 0.3], [0.12, 0, 0.3], color="black", linestyle="--", label="y=|x|")
# plt.grid()
# plt.title("Synthetic datasets containing one object", fontsize=20)
# plt.xlabel("Collision distance [m]", fontsize=20)
# plt.ylabel("Translation error [m]", fontsize=20)
# plt.legend(fontsize=15)
# plt.yticks(fontsize=15)
# plt.xticks(fontsize=15)
# plt.show()

plt.scatter(dist_megapose_ycbv[:, 0], dist_megapose_ycbv[:, 1], label="Megapose - YCB-Video", color="tab:blue", marker="o", s=150)
plt.scatter(dist_megapose_tless[:, 0], dist_megapose_tless[:, 1], label="Megapose - T-LESS", color="tab:brown", marker="o", s=150)
plt.scatter(dist_coll_tless[:, 0], dist_coll_tless[:, 1], label="Our method (Floor collisions) - T-LESS", color="tab:olive", marker="x", s=150)
plt.scatter(dist_coll_ycbv[:, 0], dist_coll_ycbv[:, 1], label="Our method (Floor collisions) - YCB-Video", color="tab:red", marker="x", s=150)
plt.grid()
plt.title("Synthetic datasets containing one object", fontsize=20)
plt.xlabel("Camera distance [m]", fontsize=20)
plt.ylabel("Collision distance [m]", fontsize=20)
plt.legend(fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()

plt.scatter(dist_megapose_ycbv[:, 0], dist_megapose_ycbv[:, 1], label="Megapose - YCB-Video", color="tab:blue", marker="o", s=150)
plt.scatter(dist_megapose_tless[:, 0], dist_megapose_tless[:, 1], label="Megapose - T-LESS", color="tab:brown", marker="o", s=150)
plt.scatter(dist_coll_grav_tless[:, 0], dist_coll_grav_tless[:, 1], label="Our method (Floor collisions + Gravity) - T-LESS", color="tab:olive", marker="x", s=150)
plt.scatter(dist_coll_grav_ycbv[:, 0], dist_coll_grav_ycbv[:, 1], label="Our method (Floor collisions + Gravity) - YCB-Video", color="tab:red", marker="x", s=150)
plt.grid()
plt.title("Synthetic datasets containing one object", fontsize=20)
plt.xlabel("Camera distance [m]", fontsize=20)
plt.ylabel("Collision distance [m]", fontsize=20)
plt.legend(fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()





############ BT ###############

err_megapose = np.array(all_err_points["Megapose"])
err_coll = np.array(all_err_points["Collisions"])
err_coll_grav = np.array(all_err_points["Collisions+Gravity"])
dist_coll = np.array(all_dist_points["Collisions"])
dist_coll_grav = np.array(all_dist_points["Collisions+Gravity"])
dist_megapose = np.array(all_dist_points["Megapose"])

# dist_err_corr = np.array(all_dist_err_corr_points["Collisions"])
# dist_err_corr_grav = np.array(all_dist_err_corr_points["Collisions+Gravity"])
# dist_err_corr_megapose = np.array(all_dist_err_corr_points["Megapose"])
# plt.scatter(dist_err_corr_grav[:, 0], dist_err_corr_grav[:, 1], label="Our method (Floor collisions + Gravity)", color="r", marker="o", s=70)
# plt.scatter(dist_err_corr_megapose[:, 0], dist_err_corr_megapose[:, 1], label="Megapose", color="b", marker="x", s=70)
# plt.grid()
# plt.title("T-LESS Synthetic", fontsize=20)
# plt.xlabel("Collision distance [m]", fontsize=20)
# plt.ylabel("Translation error [m]", fontsize=20)
# plt.legend(fontsize=20)
# plt.yticks(fontsize=15)
# plt.xticks(fontsize=15)
# plt.show()

plt.scatter(dist_coll[:, 0], dist_coll[:, 1], label="Our method (Floor collisions)", color="r", marker="o", s=70)
plt.scatter(dist_megapose[:, 0], dist_megapose[:, 1], label="Megapose", color="b", marker="x", s=70)
plt.grid()
plt.title("T-LESS Synthetic", fontsize=20)
plt.xlabel("Camera distance [m]", fontsize=20)
plt.ylabel("Collision distance [m]", fontsize=20)
plt.legend(fontsize=20)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()

plt.scatter(dist_coll_grav[:, 0], dist_coll_grav[:, 1], label="Our method (Floor collisions + Gravity)", color="r", marker="o", s=70)
plt.scatter(dist_megapose[:, 0], dist_megapose[:, 1], label="Megapose", color="b", marker="x", s=70)
plt.grid()
plt.title("T-LESS Synthetic", fontsize=15)
plt.xlabel("Camera distance [m]", fontsize=15)
plt.ylabel("Collision distance [m]", fontsize=15)
plt.legend(fontsize=15)
plt.yticks(fontsize=15)
plt.xticks(fontsize=15)
plt.show()

plt.scatter(err_coll[:, 0], err_coll[:, 1], label="Our method (Floor collisions)", color="r", marker="o"),
plt.scatter(err_megapose[:, 0], err_megapose[:, 1], label="Megapose", color="b", marker="x")
plt.grid()
plt.title("T-LESS Synthetic")
plt.xlabel("Camera distance [m]")
plt.ylabel("Translation error [m]")
plt.legend()
plt.show()


plt.scatter(err_coll_grav[:, 0], err_coll_grav[:, 1], label="Our method (Floor collisions + Gravity)", color="g", marker="o")
plt.scatter(err_megapose[:, 0], err_megapose[:, 1], label="Megapose", color="b", marker="x")
plt.grid()
plt.title("T-LESS Synthetic")
plt.xlabel("Camera distance [m]")
plt.ylabel("Translation error [m]")
plt.legend()
plt.show()


############ CORR DIST - ERROR ############

for i, err_points_label in enumerate(all_err_points):
    err_points = np.array(all_err_points[err_points_label])
    if err_points_label == "Megapose":
        plt.scatter(err_points[:, 0], err_points[:, 1], label=err_points_label, color=used_colors[i], marker="x")
    else:  
        plt.scatter(err_points[:, 0], err_points[:, 1], label=err_points_label, color=used_colors[i])
plt.grid()
plt.title("Correlation between camera distance and translation error on synthetic dataset containing one object")
plt.xlabel("Camera distance [m]")
plt.ylabel("Translation error [m]")
plt.legend()
plt.show()

############ CORR DIST - COLL ############

for i, dist_points_label in enumerate(all_dist_points):
    dist_points = np.array(all_dist_points[dist_points_label])
    if dist_points_label == "Megapose":
        plt.scatter(dist_points[:, 0], dist_points[:, 1], label=dist_points_label, color=used_colors[i], marker="x")
    else:
        plt.scatter(dist_points[:, 0], dist_points[:, 1], label=dist_points_label, color=used_colors[i])
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

############ CORR DIST - ERROR PER OBJECT ############

for i, obj_id in enumerate(obj_names):
    plt.subplot2grid((rows, cols), (i // cols, i % cols))
    for j, csv_name in enumerate(csv_vis_names):
        err_points = np.array(all_per_obj_err_points[csv_name][obj_id])
        if csv_name == "Megapose":
            plt.scatter(err_points[:, 0], err_points[:, 1], label=csv_name, color=used_colors[j], marker="x")
        else:
            plt.scatter(err_points[:, 0], err_points[:, 1], label=csv_name, color=used_colors[j])
    plt.grid()
    plt.title(f"Object {obj_id}")
    plt.xlabel("Camera distance [m]")
    plt.ylabel("Translation error [m]")
    plt.legend()
plt.suptitle("Correlation between camera distance and translation error on synthetic dataset containing one object")
plt.show()

############ CORR DIST - COLL PER OBJECT ############

for i, obj_id in enumerate(obj_names):
    plt.subplot2grid((rows, cols), (i // cols, i % cols))
    for j, csv_name in enumerate(csv_vis_names):
        dist_points = np.array(all_per_obj_dist_points[csv_name][obj_id])
        if csv_name == "Megapose":
            plt.scatter(dist_points[:, 0], dist_points[:, 1], label=csv_name, color=used_colors[j], marker="x")
        else:
            plt.scatter(dist_points[:, 0], dist_points[:, 1], label=csv_name, color=used_colors[j])
    plt.grid()
    plt.title(f"Object {obj_id}")
    plt.xlabel("Camera distance [m]")
    plt.ylabel("Collision distance [m]")
    plt.legend()
plt.suptitle("Correlation between camera distance and collision distance on synthetic dataset containing one object")
plt.show()
