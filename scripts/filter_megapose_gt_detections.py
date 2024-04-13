import json
import csv
from tqdm import tqdm
from config import POSES_OUTPUT_PATH, DATASETS_PATH

dataset_name = "hopevideo" # <=== Input
csv_input_name = f"refiner-final_{dataset_name}-test.csv" # <=== Input
csv_output_name = f"refiner-final-filtered_{dataset_name}-test_new.csv" # <=== Input

assert csv_output_name != csv_input_name

path_scenes = DATASETS_PATH / dataset_name
csv_input_path = POSES_OUTPUT_PATH / dataset_name / csv_input_name
csv_output_path = POSES_OUTPUT_PATH / dataset_name / csv_output_name

csv_lines = []
with open(csv_output_path, "w") as f:
    f.write("scene_id,im_id,obj_id,score,R,t,time\n")

def find_in_preds(path_poses, scene, im_id, obj_id, count):
    curr_count = 0
    with open(path_poses, newline="") as csvfile:
        preds_csv = csv.reader(csvfile, delimiter=',')
        for row in preds_csv:
            if row[0] == str(int(scene)) and row[1] == im_id and row[2] == str(obj_id):
                curr_count += 1
                if curr_count > 1:
                    print(f"Warning: {scene} {im_id} {obj_id} has more than one prediction")
                if curr_count == count:
                    return [row[3], row[4], row[5], row[6]]

obj_count = {}
for scene in tqdm(path_scenes.iterdir()):
    with open(scene / "scene_gt_info.json") as f:
        scene_gt_info = json.load(f)
    with open(scene / "scene_gt.json") as f:
        scene_gt = json.load(f)
    for im in scene_gt_info:
        for obj_idx in range(len(scene_gt_info[im])):
            if scene_gt_info[im][obj_idx]["visib_fract"] > 0.5:
                obj_count[f"{scene.name}_{im}_{scene_gt[im][obj_idx]['obj_id']}"] = obj_count.get(f"{scene.name}_{im}_{scene_gt[im][obj_idx]['obj_id']}", 0) + 1
                count = obj_count[f"{scene.name}_{im}_{scene_gt[im][obj_idx]['obj_id']}"]
                line = [str(int(scene.name)), im, str(scene_gt[im][obj_idx]["obj_id"]), *find_in_preds(csv_input_path, scene.name, im, scene_gt[im][obj_idx]["obj_id"], count)]
                with open(csv_output_path, "a") as f:
                    f.write(",".join([str(x) for x in line]) + "\n")