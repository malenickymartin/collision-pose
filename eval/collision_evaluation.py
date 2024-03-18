from typing import Union
import matplotlib.pyplot as plt

from eval_utils import *
from config import MESHES_DECOMP_PATH, MESHES_PATH, DATASETS_PATH, POSES_OUTPUT_PATH, FLOOR_POSES_PATH

FLOOR_MESH_PATH = Path(os.path.realpath(__file__)).parent / "data" / "floor.ply"

def eval_bproc(path_mp_outputs: Path, path_bp_gt: Path, path_meshes: Path, sw_floor_coll: bool, vis: bool):
    """
    Evaluates Blenderproc scenes with GT generated by Blenderproc (in BOP format)
    Inputs:
        path_mp_outputs: path to directory which contains N files named object_data_X.json, where N is number of images and X is image number
        path_bp_gt: path to directory which contains json files with information about specific scene (scene_gt.json and scene_camera.json)
        path_meshes: path to directory which contains subdirecotries (named with lables of mesh inside), each containing mesh and texture
        sw_floor_coll: bool value indicating whether collision with floor should be calculated
        vis: bool value indicating wherther visualisation shoud be performed
    Returns:
    """

    rigid_objects = load_meshes(path_meshes)

    # Load info
    scenes_path = list(path_mp_outputs.iterdir())
    gt_info_path = path_bp_gt / "scene_gt.json"
    gt_cam_path = path_bp_gt / "scene_camera.json"
    with open(gt_info_path, "r") as f:
        scenes_gt = json.load(f)
    with open(gt_cam_path, "r") as f:
        gt_cam = json.load(f)

    # Init errors
    err_R = 0
    err_t = 0
    num_objs = 0
    floor_coll_dist = 0
    obj_coll_dist = 0
    obj_colls = 0
    floor_colls = 0
    floor_above_dist = 0
    floor_above = 0
    num_images = 0

    corr_dist_floor = []
    corr_dist_objs = []
    corr_t = []
    corr_R = []


    # Iterate all images
    for scene_path in tqdm(scenes_path):
        num_images += 1
        scene_idx = int(scene_path.name.split("_")[-1].split(".")[0])
        with open(scene_path, "r") as f:
            scene_pred = json.load(f)
        labels = [obj["label"] for obj in scene_pred]
        scene_gt = scenes_gt[str(scene_idx)]
        se3_cam = get_se3_from_bp_cam(gt_cam[str(scene_idx)])
        # Compute collisions
        for i in range(len(labels)):
            mesh_1 = rigid_objects[labels[i]]
            se3_1 = se3_cam.inverse() * get_se3_from_mp_json(scene_pred[i])
            corr_dist_objs.append(0)
            # Collision with floor
            if sw_floor_coll:
                mesh_2 = "floor"
                se3_2 = pin.SE3.Identity()
                d = get_dist(mesh_2, mesh_1, se3_2, se3_1, visualise=vis)
                corr_dist_floor.append(abs(d))
                if d < 0:
                    floor_coll_dist -= d
                    floor_colls += 1
                elif d > 0:
                    floor_above_dist += d
                    floor_above += 1
            # Collision with other objects
            for j in range(i+1, len(labels)):
                mesh_2 = rigid_objects[labels[j]]
                se3_2 = se3_cam.inverse() * get_se3_from_mp_json(scene_pred[j])
                d = get_dist(mesh_1, mesh_2, se3_1, se3_2, visualise=vis)
                if d < 0:
                    corr_dist_objs[-1] -= d
                    obj_coll_dist -= d
                    obj_colls += 1 
            # Compare GT with predictions
            label = scene_pred[i]["label"]
            se3_pred = get_se3_from_mp_json(scene_pred[i])
            for j in range(len(scene_gt)):
                if scene_gt[j]["obj_id"] == int(label):
                    se3_gt = get_se3_from_gt(scene_gt[j])
                    break
            err_t += np.linalg.norm(se3_gt.translation - se3_pred.translation)
            err_R += np.linalg.norm(pin.log3((se3_gt.inverse() * se3_pred).rotation))
            corr_t.append(np.linalg.norm(se3_gt.translation - se3_pred.translation))
            corr_R.append(np.linalg.norm(pin.log3((se3_gt.inverse() * se3_pred).rotation)))
            num_objs += 1

    if sw_floor_coll:
        corr_floor_t_res = np.corrcoef(corr_dist_floor, corr_t)[0,1]
        corr_floor_R_res = np.corrcoef(corr_dist_floor, corr_R)[0,1]

    corr_objs_t_res = np.corrcoef(corr_dist_objs, corr_t)[0,1]
    corr_objs_R_res = np.corrcoef(corr_dist_objs, corr_R)[0,1]
    corr_res = np.corrcoef(corr_t, corr_R)[0,1]
    
    err_t /= num_objs
    err_R /= num_objs

    print("_____________  Results ______________")
    print(f"Total number of images is {len(scenes_path)}")
    print(f"Average number of objects per image is {num_objs/len(scenes_path)}")
    print("__________ Collision stats __________")
    print(f"{floor_colls+obj_colls} collision were detected, that is average of {(floor_colls+obj_colls)/num_objs} per image per object")
    print(f"{floor_colls} collisions were with floor and {obj_colls} with other objects.")
    if floor_colls > 0:
        print(f"Average depth of collision with floor is {floor_coll_dist/floor_colls}")
        print(f"Number of collisions is {floor_colls}. That is average of {floor_colls/num_images} per image.")
    if floor_above > 0:
        print(f"Average heigth above floor is {floor_above_dist/floor_above}.")
        print(f"Number times object was above floor is {floor_above}. That is average of {floor_above/num_images} per image.")
    if obj_colls > 0:
        print(f"Average depth of collision with another object is {obj_coll_dist/obj_colls}")
        print(f"Number of collisions is {obj_colls}. That is average of {obj_colls/num_images} per image.")
    print("____________ Error stats ____________")
    print(f"Average translation error is {err_t}")
    print(f"Average rotation error is {err_R}")
    print("___________  Correlation ____________")
    if sw_floor_coll:
        print(f"Correlation between translation error and distance from floor is {corr_floor_t_res}")
        print(f"Correlation between rotation error and distance from floor is {corr_floor_R_res}")
    print(f"Correlation between translation error and collision distance between objects is {corr_objs_t_res}")
    print(f"Correlation between rotation error and collision distance between objects is {corr_objs_R_res}")
    print(f"Correlation between translation error and rotation error is {corr_res}")

def eval_csv(path_scenes: Path, csv_names: list, path_csv: Path,
             path_meshes: Path, path_convex_meshes: Path,
             sw_floor_coll: bool, sw_obj_coll: bool, pre_loaded_floor: Union[None, str] = None):

    rigid_objects_decomp = load_multi_convex_meshes(path_convex_meshes)
    rigid_objects = load_meshes(path_meshes)
    
    loader = hppfcl.MeshLoader()
    floor_path = str(FLOOR_MESH_PATH)
    floor_hppfcl: hppfcl.BVHModelBase = loader.load(floor_path, scale = np.array([0.0005]*3))
    floor_hppfcl.buildConvexHull(True, "Qt")
    floor_mesh = floor_hppfcl.convex

    # Load info
    if sw_floor_coll:
        if pre_loaded_floor != None:  
            with open(pre_loaded_floor, "r") as f:
                floor_se3s = json.load(f)   
        else:
            floor_se3s = get_floor_se3s(path_scenes, rigid_objects, floor_mesh)

    all_dists = []
    all_dists_floor = []
    
    for csv_name in csv_names:
        scenes = load_csv(path_csv / csv_name)

        # Init errors
        num_objs = 0
        obj_coll_dist = 0
        obj_colls = 0
        floor_coll_dist = 0
        floor_colls = 0
        floor_above_dist = 0
        floor_above = 0

        # Other info
        num_scenes = 0
        num_images = 0
        dists = []
        dists_floor = []

        # Iterate all images
        for scene in tqdm(scenes):
            num_scenes += 1
            for im in scenes[scene]:
                if pre_loaded_floor != None and sw_floor_coll:
                    se3_floor = floor_se3s[str(scene)][str(im)]
                    se3_floor = None if se3_floor == None else pin.SE3(np.array(se3_floor["R"]), np.array(se3_floor["t"]))
                elif sw_floor_coll:
                    se3_floor = floor_se3s[scene][im]
                else: 
                    se3_floor = None
                if se3_floor == None and sw_floor_coll:
                    continue
                num_images += 1
                data = scenes[scene][im]
                labels = data["obj_id"]
                for i in range(len(labels)):
                    mesh_1 = rigid_objects[labels[i]]
                    mesh_decomp_1 = rigid_objects_decomp[labels[i]]
                    se3_1 = get_se3_from_mp_csv(data, i)
                    # Collision with floor
                    if sw_floor_coll and se3_floor != None:
                        d = get_dist(mesh_1, floor_mesh, se3_1, se3_floor)
                        dists_floor.append(d)
                        if d < 0:
                            d = get_dist_decomp(mesh_decomp_1, [floor_mesh], se3_1, se3_floor)
                            dists_floor[-1] = d
                            if d < 0:
                                floor_coll_dist -= d
                                floor_colls += 1
                        elif not scene in [48, 49, 55]: # THESE ARE THE SCENES (IN BOP YCB-V) WHERE TWO OBJECTS ARE ON TOP OF EACH OTHER
                            d = get_dist_decomp(mesh_decomp_1, [floor_mesh], se3_1, se3_floor)
                            if d > 0:
                                floor_above_dist += d
                                floor_above += 1
                    if sw_obj_coll:
                        # Collision with other objects
                        for j in range(i+1, len(labels)):
                            mesh_2 = rigid_objects[labels[j]]
                            mesh_decomp_2 = rigid_objects_decomp[labels[j]]
                            se3_2 = get_se3_from_mp_csv(data, j)
                            d = get_dist(mesh_1, mesh_2, se3_1, se3_2)
                            dists.append(d)
                            if d < 0:
                                d = get_dist_decomp(mesh_decomp_1, mesh_decomp_2, se3_1, se3_2)
                                dists[-1] = d
                                if d < 0:
                                    obj_coll_dist -= d
                                    obj_colls += 1 
                    num_objs += 1
        all_dists.append(dists)
        all_dists_floor.append(dists_floor)

        print()
        print(f"Name: {csv_name}")
        print("_____________  Results ______________")
        print(f"Total number of scenes is {num_scenes}. Total number of images is {num_images}")
        print(f"Average number of objects per image is {num_objs/num_images}")
        if sw_floor_coll:
            print("__________ Floor collision stats __________")
            if floor_colls > 0:
                print(f"Average depth of collision in between object and floor is {1000*floor_coll_dist/floor_colls} mm.")
                print(f"Number of collisions is {floor_colls}. That is average of {floor_colls/num_images} per image.")
            if floor_above > 0:
                print(f"Average heigth above floor is {floor_above_dist/floor_above}.")
                print(f"Number times object was above floor is {floor_above}. That is average of {floor_above/num_images} per image.")
        if sw_obj_coll:
            print("__________ Objects collision stats __________")
            if obj_colls > 0:
                print(f"Average depth of collision in between objects is {1000*obj_coll_dist/obj_colls} mm.")
                print(f"Number of collisions is {obj_colls}. That is average of {obj_colls/num_images} per image.")
    
    range_min = min([min(dists) for dists in all_dists])
    plt.hist(all_dists, bins=500, range=(range_min, 0.1), label=csv_names, histtype="step")
    plt.xlabel("Distance [m]")
    plt.ylabel("Frequency of occurence")
    plt.grid(True)
    plt.title("Histogram of object pair distances for HOPE-Video dataset")
    plt.legend()
    plt.show()

    range_min = min([min(dists) for dists in all_dists_floor])
    plt.hist(all_dists_floor, bins=500, range=(range_min, 0.1), label=csv_names, histtype="step")
    plt.xlabel("Distance [m]")
    plt.ylabel("Frequency of occurence")
    plt.grid(True)
    plt.title("Histogram of object-floor distances for HOPE-Video dataset")
    plt.legend()
    plt.show()

def eval_csv_floor_non_decomp():
    dataset_name = "ycbv_bop"

    dataset_path = Path("/local2/homes/malenma3/object_detection/datasets") / dataset_name

    rigid_objects = load_meshes(dataset_path / "meshes")

    loader = hppfcl.MeshLoader()
    floor_path = str(FLOOR_MESH_PATH)
    floor_hppfcl: hppfcl.BVHModelBase = loader.load(floor_path, scale = np.array([0.0005]*3))
    floor_hppfcl.buildConvexHull(True, "Qt")
    floor_mesh = floor_hppfcl.convex

    # Load info
    csv_names = ["bop_evaluation_gdrnppdet/refiner-final_ycbv-test.csv",
                 "bop_evaluation_gdrnppdet/coarse_ycbv-test.csv",
                 "bop_evaluation_gt/refiner-final_ycbv-test.csv",
                 "bop_evaluation_gt/coarse_ycbv-test.csv"]
    floor_coll_dist = 0
    floor_colls = 0
    num_objs = 0
    num_images = 0
    for csv_name in csv_names:
        scenes = load_csv(dataset_path / csv_name)
        # Iterate all images
        for scene in tqdm(scenes):
            scene_path = dataset_path / "test_data" / f"{scene:06d}"
            with open(scene_path / "scene_gt.json", "r") as f:
                scene_json = json.load(f) 
            for im in tqdm(scenes[scene]):
                num_images += 1
                data = scenes[scene][im]
                labels = data["obj_id"]
                gt_poses = {}
                for i in scene_json[f"{im}"]:
                    gt_poses[str(i["obj_id"])] = get_se3_from_gt(i)
                # Collision with floor
                plane_coefs = find_plane(im, scene_path, scene_json, rigid_objects, floor_mesh, 2)
                if plane_coefs == None:
                    print(f"Plane not found for scene {scene} image {im}")
                    continue
                se3_floor = pin.SE3(*plane_to_se3(*plane_coefs))
                for i in range(len(labels)):
                    mesh_1 = rigid_objects[labels[i]]
                    se3_1 = gt_poses[labels[i]]
                    # Collision with floor
                    d = get_dist(mesh_1, floor_mesh, se3_1, se3_floor)
                    if d < 0:
                        floor_coll_dist -= d 
                        floor_colls += 1
                    num_objs += 1
        print()
        print("_____________  Results ______________")
        print(f"Total number of images is {num_images}")
        print(f"Average number of objects per image is {num_objs/num_images}")
        print("__________ Collision stats __________")
        if floor_colls > 0:
            print(f"Average depth of collision in between object and floor is {floor_coll_dist/floor_colls}.")
            print(f"Number of collisions is {floor_colls}. That is average of {floor_colls/num_images} per image.")


def eval_csv_pose(path_scenes, csv_names, path_csv):
    
    for csv_name in csv_names:
        err_R = 0
        err_t = 0
        missed = 0
        num_scenes = 0
        num_objs = 0
        num_images = 0
        scenes = load_csv(path_csv / csv_name)
        for scene in scenes:
            num_scenes += 1
            scene_path = path_scenes / f"{scene:06d}"
            with open(scene_path / "scene_gt.json", "r") as f:
                scene_gt = json.load(f) 
            for im in scenes[scene]:
                num_images += 1
                preds = scenes[scene][im]
                gts = scene_gt[str(im)]
                for gt in gts:
                    if str(gt["obj_id"]) in preds["obj_id"]:
                        num_objs += 1
                        idx = preds["obj_id"].index(str(gt["obj_id"]))
                        err_t += np.linalg.norm(np.array(gt["cam_t_m2c"]) - preds["t"][idx]*1000)
                        err_R += np.linalg.norm(pin.log3((np.reshape(gt["cam_R_m2c"], (3,3)).T @ preds["R"][idx])))
                    else:
                        missed += 1
        err_t /= num_objs
        err_R /= num_objs
        print()
        print(f"__________ {csv_name} ____________")
        print(f"Total number of scenes is {num_scenes}, number of images is {num_images}.")
        print(f"Average number of objects per image is {num_objs/num_images}")
        print(f"Number of missed detections is {missed}")
        print(f"Average translation error is {err_t} mm")
        print(f"Average rotation error is {err_R} rad")


if __name__ == "__main__":
    eval = int(input("Select evaluation: "))
    if eval == 0:
        #EVAL COLLISIONS ON BLENDERPROC
        dataset_path = Path("eval/data/ycbv_convex_two")
        path_meshes = Path("eval/data/ycbv_convex/meshes")
        path_mp_outputs = Path("eval/data/ycbv_convex_two/happypose/outputs")
        path_bp_gt = Path("eval/data/ycbv_convex_two/train_pbr/000000")
        floor = True
        vis = False
        eval_bproc(path_mp_outputs, path_bp_gt, path_meshes, floor, vis)
    elif eval == 1:
        #EVAL COLLISIONS ON BOP (MEDERICS MEGAPOSE INFERENCE OF BOP YCBV)
        ds_name = "hopevideo" # <= INPUT
        floor_name = "hope_bop_floor_poses_1mm_res_optimized.json" #None # or string # <= INPUT
        csv_names = ["refiner-final-filtered_hopevideo-test.csv",
                     "filtered_fixed_derivative/05-001-09-50-001-006-026-optimized_hopevideo-test.csv"] # <= INPUT

        path_scenes = DATASETS_PATH / ds_name
        path_csv = POSES_OUTPUT_PATH / ds_name
        path_meshes = MESHES_PATH / ds_name
        path_convex_meshes = MESHES_DECOMP_PATH / ds_name
        floor_se3_path = FLOOR_POSES_PATH / floor_name if floor_name != None else None
        eval_csv(path_scenes, csv_names, path_csv, path_meshes, path_convex_meshes, True, True, floor_se3_path)
    elif eval == 2:
        #EVAL POSE ERRORS ON BOP (ONLY FOR DATASETS WITH ONE LABEL PER IMAGE, ELSE USE BOP TOOLKIT)
        ds_name = "ycbv" # <= INPUT
        csv_names = ["gt-refiner-final_ycbv-test.csv"] # <= INPUT

        path_scenes = DATASETS_PATH / ds_name
        path_csv = POSES_OUTPUT_PATH / ds_name
        eval_csv_pose(path_scenes, csv_names, path_csv)
    elif eval == 3:
        eval_csv_floor_non_decomp()