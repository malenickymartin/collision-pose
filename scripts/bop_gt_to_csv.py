import json
from eval.eval_utils import get_se3_from_gt
from config import DATASETS_PATH, POSES_OUTPUT_PATH


def gt_to_csv():
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


def create_test_targets():
    dataset_name = input("Enter the name of the dataset: ")

    test_targets = []
    for scene in (DATASETS_PATH / dataset_name).iterdir():
        with open(scene / "scene_gt.json", "r") as f:
            scene_gt = json.load(f)
        for im in scene_gt:
            for obj in scene_gt[im]:
                obj_id = obj["obj_id"]
                scene_id = int(scene.name)
                im_id = int(im)
                if any([d["im_id"] == im_id and d["obj_id"] == obj_id and d["scene_id"] == scene_id 
                        for d in test_targets]):
                    for d in test_targets:
                        if d["im_id"] == im_id and d["obj_id"] == obj_id and d["scene_id"] == scene_id:
                            d["inst_count"] += 1
                else:
                    test_targets.append({"im_id": im_id, "inst_count": 1, "obj_id": obj_id, "scene_id": scene_id})
    
    with open(DATASETS_PATH / dataset_name / "test_targets_bop19.json", "w") as f:
        json.dump(test_targets, f, indent=2)

gt_to_csv()