from config import DATASETS_PATH, POSES_OUTPUT_PATH, FLOOR_POSES_PATH
import pinocchio as pin
import numpy as np
import json
import csv

def json_2_csv(dataset_name, scene_path):
    csv_output_name = f"refiner-final_{dataset_name}-test.csv" # <=== Input

    jsons_path = scene_path / dataset_name / "happypose" / "outputs"
    csv_path = POSES_OUTPUT_PATH / dataset_name / csv_output_name

    with open(csv_path, "w") as f:
        f.write("scene_id,im_id,obj_id,score,R,t,time\n")

    for mp_out in jsons_path.iterdir():
        with open(mp_out, "r") as f:
            mp_out_data = json.load(f)
        scene_id = "0"
        im_id = mp_out.name.split("_")[2].split(".")[0]
        for obj in mp_out_data:
            obj_id = str(obj["label"])
            se3 = pin.XYZQUATToSE3(np.concatenate([obj["TWO"][1], obj["TWO"][0]]))
            R = " ".join(str(item) for item in se3.rotation.reshape(9).tolist())
            t = " ".join(str(item) for item in (se3.translation*1000).tolist())
            line = [scene_id, im_id, obj_id, "1", R, t, "-1"]
            with open(csv_path, "a") as f:
                f.write(",".join([str(x) for x in line]) + "\n")

def scene_gt_labels_change(dataset_name, scene_path):
    scene_gt_path = scene_path / dataset_name / "train_pbr/000000" /"scene_gt.json"
    scene_gt_new_path = scene_path / dataset_name / "train_pbr/000000" / "scene_gt_new.json"

    with open(scene_gt_path, "r") as f:
        scene_gt_data = json.load(f)
    for scene in scene_gt_data:
        for obj in scene_gt_data[scene]:
            obj["obj_id"] = int(obj["obj_id"])
    with open(scene_gt_new_path, "w") as f:
        json.dump(scene_gt_data, f)

def save_floor_poses(dataset_name, scene_path):
    cam_gt_path = scene_path / dataset_name / "train_pbr/000000" / "scene_camera.json"
    floor_pose_path = FLOOR_POSES_PATH / f"{dataset_name}_synt_floor_gt.json"

    with open(cam_gt_path, "r") as f:
        cam_gt_data = json.load(f)
    floor_poses = {"0":{}}
    for im in cam_gt_data:
        floor_poses["0"][im] = {}
        floor_poses["0"][im]["R"] = np.reshape(cam_gt_data[im]["cam_R_w2c"], (3,3)).tolist()
        floor_poses["0"][im]["t"] = (np.array(cam_gt_data[im]["cam_t_w2c"])/1000).tolist()
    with open(floor_pose_path, "w") as f:
        json.dump(floor_poses, f)

dataset_name = "tlessone"
scene_path = DATASETS_PATH / dataset_name
json_2_csv(dataset_name, scene_path)
save_floor_poses(dataset_name, scene_path)