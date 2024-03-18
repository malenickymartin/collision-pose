import json
import csv
from tqdm import tqdm
from config import POSES_OUTPUT_PATH, DATASETS_PATH

dataset_name = "tless" # <=== Input
csv_input_name = f"refiner-final_{dataset_name}-test.csv" # <=== Input
csv_output_name = f"refiner-final-filtered_{dataset_name}-test.csv" # <=== Input

assert csv_output_name != csv_input_name

path_scenes = DATASETS_PATH / dataset_name
csv_input_path = POSES_OUTPUT_PATH / dataset_name / csv_input_name
csv_output_path = POSES_OUTPUT_PATH / dataset_name / csv_output_name

csv_lines = []
with open(csv_output_path, "w") as f:
    f.write("scene_id,im_id,obj_id,score,R,t,time\n")

def find_in_preds(path_poses, scene, im_id, obj_id):
    with open(path_poses, newline="") as csvfile:
        preds_csv = csv.reader(csvfile, delimiter=',')
        for row in preds_csv:
            if row[0] == str(int(scene)) and row[1] == im_id and row[2] == str(obj_id):
                return [row[3], row[4], row[5], row[6]]

for scene in tqdm(path_scenes.iterdir()):
    with open(scene / "scene_gt_info.json") as f:
        scene_gt_info = json.load(f)
    with open(scene / "scene_gt.json") as f:
        scene_gt = json.load(f)
    for im in scene_gt_info:
        for obj_idx in range(len(scene_gt_info[im])):
            if scene_gt_info[im][obj_idx]["visib_fract"] > 0.5:
                line = [str(int(scene.name)), im, str(scene_gt[im][obj_idx]["obj_id"]), *find_in_preds(csv_input_path, scene.name, im, scene_gt[im][obj_idx]["obj_id"])]
                with open(csv_output_path, "a") as f:
                    f.write(",".join([str(x) for x in line]) + "\n")