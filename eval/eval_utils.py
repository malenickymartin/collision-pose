from sklearn import linear_model
from PIL import Image
import argparse
import numpy as np
import os
import json
from tqdm import tqdm
import csv
import pinocchio as pin

import pydiffcol
from pathlib import Path
import hppfcl
from pydiffcol.utils import add_arguments_to_parser, select_strategy, select_targets, bring_shapes_to_dist
from pydiffcol.utils_render import create_visualizer, draw_scene, renderPoint, draw_shape

FLOOR_MESH_PATH = Path(os.path.realpath(__file__)).parent / "data" / "floor.ply"

def get_dist(path_1, path_2, SE3_1, SE3_2, scale_1=1, scale_2=1, visualise=False, bring_to_zero=False):
    """
    Calculates the distance between two meshes.
    Inputs:
        path_1: is location of mesh (str or Path) || string "floor" which will load large square with pre defined location || already loaded hppfcl.ShapeBase object
        path_2: same as path_1
        SE3_1: is list [[3x3],[3x1]] where 3x3 is rotation matrix and 3x1 if translation vector (both are numpy arrays) || pinocchio SE3 object
        SE3_2: same as SE3_1
        scale_1: is float scale which will convert the units of the loaded mesh to meters 
                 (note that the load_meshes function already converts the units, so this will usually be 1)
        scale_2: same as scale_1
        visualise: is a bool indicating whether the visualisation will be created 
    Returns: 
        d: distance between the two objects in meters
    """

    if isinstance(SE3_1, list):
        assert len(SE3_1) == len(SE3_2) == 2, "SE3 must be array of rotation and translation '(rot, trans)'"
        assert SE3_1[0].shape == SE3_2[0].shape == (3,3), "Rotation part of SE3 is not 3x3 numpy array"
        assert SE3_1[1].shape == SE3_2[1].shape == (3,), "Translation part of SE3 is not 3x1 numpy array"
    else:
        assert isinstance(SE3_1, pin.SE3) and isinstance(SE3_2, pin.SE3), "SE3 in not numpy array nor pin SE3"

    loader = hppfcl.MeshLoader()

    if isinstance(path_1, (Path, str)):
        path_1 = str(path_1)
        if path_1 == "floor":
            path_1 = str(FLOOR_MESH_PATH)
        mesh1: hppfcl.BVHModelBase = loader.load(path_1, scale = np.array([scale_1]*3))
        mesh1.buildConvexHull(True, "Qt")
        shape1 = mesh1.convex
    elif isinstance(path_1, hppfcl.ShapeBase) or isinstance(path_1, hppfcl.BVHModelOBBRSS) :
        shape1 = path_1
    else:
        raise Exception("path_1 is not Path, string nor a Convex boject.")

    if isinstance(path_2, (Path, str)):
        path_2 = str(path_2)
        if path_2 == "floor":
            path_2 = str(FLOOR_MESH_PATH)
        mesh2: hppfcl.BVHModelBase = loader.load(path_2, scale=np.array([scale_2]*3))
        mesh2.buildConvexHull(True, "Qt")
        shape2 = mesh2.convex
    elif isinstance(path_2, hppfcl.ShapeBase) or isinstance(path_1, hppfcl.BVHModelOBBRSS) :
        shape2 = path_2
    else:
        raise Exception("path_2 is not Path, string nor a Convex boject.")

    parser = argparse.ArgumentParser()
    add_arguments_to_parser(parser)
    args = parser.parse_args()
    # Only the following args are used, values are the default ones
    args.strategy = "finite_differences"
    args.max_neighbors_search_level = 1
    args.noise = 1e-3
    args.num_samples = 6

    (col_req, col_req_diff) = select_strategy(args, verbose=False)
    col_res = pydiffcol.DistanceResult()
    if not isinstance(SE3_1, pin.SE3):
        M1 = pin.SE3(*SE3_1)
    else:
        M1 = SE3_1
    if not isinstance(SE3_2, pin.SE3):
        M2 = pin.SE3(*SE3_2)
    else:
        M2 = SE3_2

    if visualise:
        p1_desired, p2_desired = select_targets(shape1, shape2)
        vis = create_visualizer(axes = True)
        draw_scene(vis, shape1, M1, shape2, M2,
            col_res.w1, col_res.w2, p1_desired, p2_desired,
            render_faces=[False, False])
        input("Press any key to continue.")
    d = pydiffcol.distance(shape1, M1, shape2, M2, col_req, col_res)
    if bring_to_zero:
        bring_shapes_to_dist(shape1, M1, shape2, M2, 0.0, col_req, col_res)
    
    if visualise:
        p1_desired, p2_desired = select_targets(shape1, shape2)
        vis = create_visualizer(axes = True)
        draw_scene(vis, shape1, M1, shape2, M2,
            col_res.w1, col_res.w2, p1_desired, p2_desired,
            render_faces=[False, False])
    
    return d


def get_dist_decomp(shape_1, shape_2, M1, M2):
    """
    Calculates the distance between two decomposed meshes.
    Inputs:
        shape_1: list of hppfcl.ShapeBase objects
        shape_2: same as shape_1
        M1: pose of the first object (pinocchio.SE3)
        M2: pose of the second object (pinocchio.SE3)
    Returns: 
        min_d: the smallest distance between the two decomposed objects in meters
    """

    parser = argparse.ArgumentParser()
    add_arguments_to_parser(parser)
    args = parser.parse_args()

    (col_req, col_req_diff) = select_strategy(args, verbose=False)
    col_res = pydiffcol.DistanceResult()

    min_dist = np.infty
    for s1 in shape_1:
        for s2 in shape_2:
            d = pydiffcol.distance(s1, M1, s2, M2, col_req, col_res)
            min_dist = min(min_dist, d)

    return min_dist


def get_se3_from_mp_json(pred: dict):
    """
    Calculates pin.SE3 pose of object predicted by Megapose. The pose consists of orthogonal rotation matrix and translation in meters.
    Inputs:
        pred: Information about one object from given scene and given image generated by Megapose in json.
    Returns:
        se3: pinocchio SE3 transform describing the object pose
    """

    t = pred["TWO"][1]
    quat = pred["TWO"][0]
    norm = np.linalg.norm(quat)
    quat_norm = quat/norm
    xyzquat = t + list(quat_norm)
    se3 = pin.XYZQUATToSE3(xyzquat)
    return se3


def get_se3_from_gt(pred: dict):
    """
    Calculates pin.SE3 pose of object in Blenderproc/BOP scene. The pose consists of orthogonal rotation matrix and translation in meters.
    Inputs:
        pred: Part of json file with info about specific image and object. The json is from BOP or generated by Blenderproc in BOP format.
    Returns:
        se3: pinocchio SE3 transform describing the camera pose
    """

    t = np.array(pred["cam_t_m2c"])/1000
    R = np.reshape(pred["cam_R_m2c"], (3,3))

    U,_,V = np.linalg.svd(R)
    R_norm = U @ V
    se3 = pin.SE3(R_norm, t)
    return se3


def get_se3_from_bp_cam(gt_cam: dict):
    """
    Calculates pin.SE3 pose of camera in Blenderproc scene. The pose consists of orthogonal rotation matrix and translation in meters.
    Inputs:
        gt_cam: Loaded json file with information about the camera for given scene and given image generated by Blenderproc in BOP format
    Returns:
        se3: pinocchio SE3 transform describing the camera pose
    """
    t = np.array(gt_cam["cam_t_w2c"])/1000
    R = np.reshape(gt_cam["cam_R_w2c"], (3,3))
    U,_,V = np.linalg.svd(R)
    R_norm = U @ V
    se3 = pin.SE3(R_norm, t)
    return se3


def get_se3_from_mp_csv(data: dict, i: int):
    """
    Calculates pin.SE3 pose of object predicted by Megapose. The pose consists of orthogonal rotation matrix and translation in meters.
    Inputs:
        data: dict with information about image, see load_csv
        i: id of object
    Returns:
        se3: pinocchio SE3 transform describing the object pose
    """

    R = data["R"][i]
    t = data["t"][i]
    U,_,V = np.linalg.svd(R)
    R_norm = U @ V
    se3 = pin.SE3(R_norm, t)
    return se3


def load_csv(mp_pred_path):
    """
    Parses the CSV output of Megapose into dictionary of scences, where each scene 
    Inputs:
        mp_pred_path: Path to the CSV file generated by Megapose
    Returns:
        scenes: three dicts inside each other, the first layer dict keys are idxs of scenes (int), the second layer dict keys are images idxs and the third layer 
                has keys either "R", "t" or "obj_id", the values for those keys are respectively list of rotation matrices (list of 3x3 np.ndarray),
                list of translation vectors (list of 3x1 np.ndarray) and list of object ids (list of strings), where index of the list corresponds to one object
    """

    with open(mp_pred_path, newline="") as csvfile:
        preds_csv = csv.reader(csvfile, delimiter=',')
        scenes = {}
        for i, pred in enumerate(preds_csv):
            if i == 0:
                continue
            scene_id = int(pred[0])
            im_id = int(pred[1])
            obj_id = pred[2]
            R = np.reshape([float(n) for n in (pred[4].split(" "))], (3,3))
            t = np.array([float(n) for n in (pred[5].split(" "))])/1000
            if scene_id in scenes:
                if im_id in scenes[scene_id]:
                    scenes[scene_id][im_id]["R"].append(R)
                    scenes[scene_id][im_id]["t"].append(t)
                    scenes[scene_id][im_id]["obj_id"].append(obj_id)
                else:
                    scenes[scene_id][im_id] = {"R":[R], "t":[t], "obj_id":[obj_id]}
            else:
                scenes[scene_id] = {im_id:{"R":[R], "t":[t], "obj_id":[obj_id]}}
    return scenes


def load_meshes(dataset_path, mesh_units = 0.001):
    """
    Creates a dataset of rigid objects.
    Inputs:
        dataset_path: path to the directory with meshes (pathlib.Path), each mesh should be in its own directory (label of the object will be the directory name),
                      the directory should contain the mesh and texture 
        mesh_units: scale which will convert the units of the loaded mesh to meters (float)
    Returns:
        rigid_objects: dict where keys are labels of meshes and values are convex hulls of the meshes
    """

    loader = hppfcl.MeshLoader()
    rigid_objects = {}
    object_dirs = (dataset_path).iterdir()
    for object_dir in object_dirs:
        print(f"Loading model {object_dir.name}")
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        mesh: hppfcl.BVHModelBase = loader.load(str(mesh_path), scale=np.array([mesh_units]*3))
        mesh.buildConvexHull(True, "Qt")
        shape = mesh.convex
        rigid_objects[label] = shape
        print(f"Model {object_dir.name} loaded.")
    return rigid_objects


def load_multi_convex_meshes(dataset_path, mesh_units = 0.001):
    """
    Creates a dataset of convex decomposition of rigid objects.
    Inputs:
        dataset_path: path to the directory with meshes (pathlib.Path), each mesh should be in its own directory (label of the object will be the directory name),
                      the directory should contain the mesh and texture 
        mesh_units: scale which will convert the units of the loaded mesh to meters (float)
    Returns:
        rigid_objects: dict where keys are labels of meshes and values are lists of convex decomposition of the meshes
    """

    loader = hppfcl.MeshLoader()
    rigid_objects = {}
    object_dirs = (dataset_path).iterdir()
    for object_dir in tqdm(object_dirs):
        label = object_dir.name
        new_mesh = []
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                mesh: hppfcl.BVHModelBase = loader.load(str(fn), scale=np.array([mesh_units]*3))
                mesh.buildConvexHull(True, "Qt")
                shape = mesh.convex
                new_mesh.append(shape)
        assert len(new_mesh) > 0, f"No mehses found for object {label}."
        rigid_objects[label] = new_mesh
    return rigid_objects


def img_2_world(u, K) -> np.ndarray:
    """
    Transforms pixel position to world position.
    Inputs:
        u: is list [a,b,z], where a is pixel row, b is pixel column, z is distance
        K: is camera matrix [[fx,0,cx],[0,fy,cy],[0,0,1]]
    Returns:
        vec: is vector from camera to object on pixel u (np.ndarray([x,y,z]))
    """

    z = u[2]
    x = (u[0]-K[0,2])*(z/K[0,0])
    y = (u[1]-K[1,2])*(z/K[1,1])
    vec = np.array([x,y,z])
    return vec


def plane_to_se3(a: float, b: float, d: float):
    """
    Transforms pixel position to world position.
    Inputs:
        a,b,d: are coeficients of plane z = ax + by + d
    Returns:
        R_CR: is rotation from camera to the plane
        t_CR: is translation from camera to the plane
    """

    n = np.array([a,b,-1])
    r_z = n/np.linalg.norm(n)
    v_x = np.array([1,1,a+b])
    r_x = v_x/np.linalg.norm(v_x)
    r_y = np.cross(r_z, r_x)/np.linalg.norm(np.cross(r_z, r_x))

    R_CR = np.array([r_x, r_y, r_z]).transpose()
    t_CR = np.array([0,0,d/1000])

    return [R_CR, t_CR]


def fit_plane(X: np.ndarray, y: np.ndarray):
    """
    Uses RANSAC to fit plane to point cloud.
    Inputs:
        X: Nx2 list of points wrt camera, where N is number of points
        y: list of length N, with distances of points from camera plane
    Returns:
        a,b,d: are coeficients of plane z = ax + by + d
        ransac.inlier_mask_: boolean mask of inliers classified as True
    """

    ransac = linear_model.RANSACRegressor(residual_threshold=10, min_samples=5, max_trials=10000, stop_probability=0.995)
    ransac.fit(X, y)
    d,a,b = ransac.estimator_.intercept_, ransac.estimator_.coef_[0], ransac.estimator_.coef_[1]
    return a, b, d, ransac.inlier_mask_


def find_plane(im, scene_path, scene_json, rigid_objects, floor_mesh, step):
    """
    Uses RANSAC to fit plane to point cloud. Returns the plane coeficients only if all objects in rigid_objects are close to the fitted plane, otherwise
    the plane is fitted again with outlier points.
    Inputs:
        im: image number (int)
        scene_path: path to the scene directory (pathlib.Path)
        scene_json: opened json file contaning info about the scene in BOP format
        rigid_objects: dict of rigid_objects, see load_meshes function
        floor_mesh: large box mesh used for calculating distances between floor and objects (hppfcl.ShapeBase)
        step: every which point from point cloud to account with
    Returns:
        a,b,d: are coeficients of plane z = ax + by + d or None if plane was not found
    """

    with open(scene_path / "scene_camera.json", "r") as f:
        camera_json = json.load(f)
    depth = Image.open(scene_path / "depth" / f"{im:06d}.png")
    depth = np.array(depth)*camera_json[f"{im}"]["depth_scale"]
    K = np.reshape(camera_json[f"{im}"]["cam_K"], (3,3))
    gt_poses = {}
    for i in scene_json[f"{im}"]:
        gt_poses[str(i["obj_id"])] = get_se3_from_gt(i)

    Xy = []
    rows = np.arange(0, depth.shape[0], step=step, dtype=np.int32)
    columns = np.arange(0, depth.shape[1], step=step, dtype=np.int32)
    for i in rows:
        for j in columns:
            if depth[i,j] > 1e-3:
                Xy.append(img_2_world([j,i,depth[i,j]], K))
    Xy = np.array(Xy)
    X = Xy[:,:2]
    y = Xy[:,2]

    num_fits = 7

    for fit in range(num_fits):
        a, b, c, inlier_mask = fit_plane(X, y)
        outlier_mask = np.logical_not(inlier_mask)
        se3_floor = pin.SE3(*plane_to_se3(a,b,c))
        dist_sum = 0
        one_allowed_mistake = False
        for i in gt_poses:
            d = get_dist(floor_mesh, rigid_objects[i], se3_floor, gt_poses[i])
            if d > 0.05 and d < 1 and one_allowed_mistake == False:
                one_allowed_mistake = True
            else:
                dist_sum += abs(d)
        if dist_sum < 0.1:
            return [a,b,c]
        else:
            X = X[outlier_mask]
            y = y[outlier_mask]
    
    if fit >= num_fits-1:
        return None
    

def get_floor_se3s(scenes_path, rigid_objects, floor_mesh):
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

    floor_se3s = {}
    for scene_path in tqdm(scenes_path.iterdir()):
        scene = int(scene_path.name)
        floor_se3s[scene] = {}
        scene_path = scenes_path / f"{scene:06d}"
        with open(scene_path / "scene_gt.json", "r") as f:
            scene_json = json.load(f)
        for im_str in tqdm(scene_json):
            im = int(im_str)
            plane_coefs = find_plane(im, scene_path, scene_json, rigid_objects, floor_mesh, 2)
            if plane_coefs == None:
                print(f"Plane not found for scene {scene} image {im}")
                se3_floor = None
            else:
                se3_floor = pin.SE3(*plane_to_se3(*plane_coefs))
            floor_se3s[scene][im] = se3_floor
    return floor_se3s


def draw_pc(vis, depth, K, se3_floor):
    """
    Draws point cloud and floor plane of scene.
    Inputs:
        vis: meshcat visualizer
        depth: loaded depth map
        K: camera matrix
        se3_floor: pose of the floor
    Returns:
    """

    Xy = []
    for i in range(depth.shape[0]):
        for j in range(depth.shape[1]):
            if depth[i,j] > 1e-3:
                Xy.append(img_2_world([j,i,depth[i,j]], K))
    Xy = np.array(Xy)
    Xy /= 1000
    for i in range(len(Xy)):
        if i % 50 != 0:
            continue
        se3 = pin.SE3.Identity()
        se3.translation = Xy[i]
        renderPoint(vis, Xy[i], f"point_{i}", color=np.array([144, 169, 183, 255]) / 255, radius_point=3e-3)

    loader = hppfcl.MeshLoader()
    path = str(FLOOR_MESH_PATH)
    mesh: hppfcl.BVHModelBase = loader.load(path, scale=np.array([1]*3))
    mesh.buildConvexHull(True, "Qt")
    shape = mesh.convex
    draw_shape(vis, shape, "plane", se3_floor, np.array([170, 236, 149, 255]) / 255, render_faces=False)