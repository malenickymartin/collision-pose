import json
from eval.eval_utils import load_meshes, load_mesh, draw_shape, get_se3_from_bp_cam, get_se3_from_gt
from tqdm import tqdm
import pinocchio as pin
from config import FLOOR_POSES_PATH, FLOOR_MESH_PATH, DATASETS_PATH, MESHES_PATH
import hppfcl
from pydiffcol.utils_render import create_visualizer
import numpy as np

def draw_pc_and_objects(vis, floor_se3, floor_mesh_vis, rigid_objects_vis, scene_json):
    vis.delete()
    draw_shape(vis, floor_mesh_vis, "floor", floor_se3, np.array([110, 250, 90, 125]) / 255)
    for obj in scene_json:
        obj_id = str(obj["obj_id"])
        se3 = get_se3_from_gt(obj)
        draw_shape(vis, rigid_objects_vis[obj_id], obj_id, se3, np.array([90, 110, 250, 125]) / 255)
    


def get_floor_se3s(scenes_path, rigid_objects_vis, floor_mesh_vis, load_name, save_name):
    """
    Uses RANSAC to fit plane to point cloud for all scenes and images in dataset.
    Inputs:
        scenes_path: path (pathlib.Path) to the directory containing scenes in BOP format
        rigid_objects: dict of rigid_objects, see load_meshes function
        floor_mesh: large box mesh used for calculating distances between floor and objects (hppfcl.ShapeBase)
    Returns:
        floor_se3s: two directories inside each other, keys of the first one are scenes idxs (int) the keys of the second one are image idxs, the values are 
                    poses of the floor (pinocchio.SE3)
    """
    last_gt = {"4":427, "13":431, "9":431, "5":423, "2":423, "8":425, "10":431, "12":423, "1":425, "3":428, "6":426, "7":428, "18":415, "11":431}

    use_im = {"4": [438, 448, 455, 462, 469, 479],
        "13": [441, 450, 462, 478, 486, 500],
        "9": [437, 451, 465, 477, 484, 490, 495],
        "5": [436, 450, 458, 463, 468, 475, 493],
        "2": [434, 439, 446, 455, 459, 463, 470, 478, 489, 494],
        "8": [432, 441, 451, 459, 465, 471, 479, 484, 495],
        "10": [441, 450, 457, 465, 475, 482, 496],
        "12": [438, 455, 471, 482, 490],
        "1": [434, 442, 455, 464, 474, 488, 493],
        "3": [432, 441, 446, 454, 459, 466, 476, 493],
        "6": [434, 442, 462, 469, 475, 485, 496, 501],
        "7": [435, 447, 450, 458, 468, 478, 486, 493],
        "18": [426, 439, 452, 461, 466, 473, 480, 487, 491, 496],
        "11": [441, 452, 459, 469, 476, 486, 497]}
    vis = create_visualizer(grid=True, axes=True)
    with open(FLOOR_POSES_PATH / load_name, "r") as f:
        floor_se3s = json.load(f)

    for scene_path in tqdm(scenes_path.iterdir()):
        scene = str(int(scene_path.name))
        if not scene in use_im or scene in ["4", "13", "9", "5", "2", "8", "10", "12", "1", "3", "6", "7"]:
            continue
        print(f"Processing scene {scene}")
        scene_path = scenes_path / f"{int(scene):06d}"
        with open(scene_path / "scene_gt.json", "r") as f:
            scene_json = json.load(f)
        with open(scene_path / "scene_camera.json", "r") as f:
            scene_cam = json.load(f)
        for im_str in scene_json:
            im = int(im_str)
            if not im in use_im[scene]:
                continue
            last_im = str(last_gt[scene])
            se3_cam_last = pin.SE3(get_se3_from_bp_cam(scene_cam[last_im]))
            se3_cam_curr = pin.SE3(get_se3_from_bp_cam(scene_cam[im_str]))
            se3_floor_last = floor_se3s[scene][last_im]
            se3_floor_last = pin.SE3(np.array(se3_floor_last["R"]).reshape((3,3)), np.array(se3_floor_last["t"]))
            se3_floor = se3_cam_curr * se3_cam_last.inverse() * se3_floor_last
            draw_pc_and_objects(vis, se3_floor, floor_mesh_vis, rigid_objects_vis, scene_json[im_str])
            se3_floor = {"R":se3_floor.rotation.tolist(), "t":se3_floor.translation.tolist()}
            floor_se3s[scene][im_str] = se3_floor
            input("Press Enter to continue")

        with open(FLOOR_POSES_PATH / save_name, "w") as f:
            json.dump(floor_se3s, f)
        print(f"Saved scene {scene}")
        
    return floor_se3s

if __name__ == "__main__":
    ds_name = "tless"
    scenes_path = DATASETS_PATH / ds_name
    rigid_objects_vis = load_meshes(MESHES_PATH / ds_name, convex=False)
    floor_poses_in = "tless_bop_floor_poses_1mm_res_dilation_optimized_del_man.json"
    floor_poses_out = "tless_bop_floor_poses_1mm_res_dilation_optimized_del_man.json"
    floor_mesh_vis = load_mesh(FLOOR_MESH_PATH, 0.01, False)
    get_floor_se3s(scenes_path, rigid_objects_vis, floor_mesh_vis, floor_poses_in, floor_poses_out)