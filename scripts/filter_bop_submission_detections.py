"""
BOP Submissions may consist of multiple detections per object instance. 
This script filters the detections to only keep the highest scoring detection per object instance using BOPs test_targets_bop19.json.
"""

import json
from pathlib import Path
import csv

from config import POSES_OUTPUT_PATH

from eval.eval_utils import load_csv


dataset_name = "ycbv" # <=== Input
csvs_dir = "bop_submissions" # <=== Input
input_csv_name = f"hcceposebf_ycbv-test.csv" # <=== Input

output_csv_name = input_csv_name.split("_")[0] + "-filtered_" + input_csv_name.split("_")[1]
bop_path = Path("/local2/homes/malenma3/bop_toolkit")
scenes = load_csv(POSES_OUTPUT_PATH / dataset_name / csvs_dir/ input_csv_name)
targets_json = json.load(open(bop_path / "bop_data" / dataset_name / "test_targets_bop19.json"))

def targets_to_tree(targets):
    tree = {}
    for target in targets:
        scene_id = target["scene_id"]
        im_id = target["im_id"]
        obj_id = target["obj_id"]
        inst_count = target["inst_count"]
        if scene_id not in tree:
            tree[scene_id] = {}
        if im_id not in tree[scene_id]:
            tree[scene_id][im_id] = {}
        tree[scene_id][im_id][obj_id] = inst_count
    return tree

def n_best_predictions(n, target_obj_id, image):
    id_scores = []
    for i, obj_id in enumerate(image["obj_id"]):
        if int(obj_id) == target_obj_id:
            id_scores.append((i, image["score"]))
    if len(id_scores) == 0:
        return None
    id_scores.sort(key=lambda x: x[1], reverse=True)
    image_new = {key: [] for key in image}
    for key in image:
        for i in range(n):
            image_new[key].append(image[key][id_scores[i][0]])
    return image_new

scenes_new = {}
targets = targets_to_tree(targets_json)
for scene in scenes:
    scenes_new[scene] = {}
    for image in scenes[scene]:
        scenes_new[scene][image] = {k : [] for k in scenes[scene][image]}
        for obj_id in targets[scene][image]:
            image_new = n_best_predictions(targets[scene][image][obj_id], obj_id, scenes[scene][image])
            if image_new is not None:
                for key in image_new:
                    scenes_new[scene][image][key].extend(image_new[key])

with open(POSES_OUTPUT_PATH / dataset_name / csvs_dir / output_csv_name, "w") as f:
    f.write("scene_id,im_id,obj_id,score,R,t,time\n")
    for scene in scenes_new:
        for image in scenes_new[scene]:
            for i in range(len(scenes_new[scene][image]["obj_id"])):
                f.write(f"{scene},{image},{scenes_new[scene][image]['obj_id'][i]},{scenes_new[scene][image]['score'][i]},"
                        f"{' '.join(str(item) for item in scenes_new[scene][image]['R'][i].reshape(9).tolist())},"
                        f"{' '.join(str(item) for item in (scenes_new[scene][image]['t'][i]*1000).tolist())},"
                        f"{scenes_new[scene][image]['time'][i]}\n")