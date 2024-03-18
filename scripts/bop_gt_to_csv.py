import json
from eval.eval_utils import get_se3_from_gt
from config import DATASETS_PATH, POSES_OUTPUT_PATH


dataset_name = input("Enter the name of the dataset: ")

output_csv_name = f"gt_{dataset_name}-test.csv"
with open(POSES_OUTPUT_PATH / dataset_name / output_csv_name, "w") as f:
    f.write("scene_id,im_id,obj_id,score,R,t,time\n")
for scene in (DATASETS_PATH / dataset_name).iterdir():
    with open(scene / "scene_gt.json", "r") as f:
        scene_gt = json.load(f)
    for im in scene_gt:
        for obj in scene_gt[im]:
            X = get_se3_from_gt(obj)
            R = " ".join(str(item) for item in X.rotation.reshape(9).tolist())
            t = " ".join(str(item) for item in (X.translation*1000).tolist())
            label = obj["obj_id"]
            scene_id = int(scene.name)
            csv_line = [scene_id, im, label, 1.0, R, t, -1]
            with open(POSES_OUTPUT_PATH / dataset_name / output_csv_name, "a") as f:
                f.write(",".join([str(x) for x in csv_line]) + "\n")