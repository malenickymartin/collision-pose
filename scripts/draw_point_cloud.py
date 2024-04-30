import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf
from robomeshcat import Scene, Object
import json
import pinocchio as pin
import numpy as np
from scipy import ndimage
import hppfcl
from PIL import Image

from src.vis import draw_shape
from eval.eval_utils import load_meshes, img_2_world
from config import MESHES_PATH, FLOOR_MESH_PATH, FLOOR_POSES_PATH, DATASETS_PATH


def load_shapes_robomeshcat(meshes_ds_name, rigid_objects, mesh_units = 0.001):
    rigid_objects_rmc = {}
    for l in rigid_objects.keys():
        obj_dir = MESHES_PATH / meshes_ds_name / l
        mesh_path = obj_dir /f"obj_{int(l):06d}.ply"
        texture_path = obj_dir /f"obj_{int(l):06d}.png"
        o = Object.create_mesh(
            mesh_path,
            scale=mesh_units,
            texture=texture_path,
            color=[1] * 3,
            )
        rigid_objects_rmc[l] = o
    o_floor = Object.create_mesh(
        FLOOR_MESH_PATH.parent / "floor" / "floor.obj",
        scale=mesh_units,
        texture=FLOOR_MESH_PATH.parent / "floor" / "Wood001_2K-JPG_Color.jpg",
        color=[0.17254902, 0.62745098, 0.17254902],
    )
    #scene_rmc.add_object(o_floor)
    return rigid_objects_rmc, o_floor

def rgbToHex(color):
    if len(color) == 4:
        c = color[:3]
        opacity = color[3]
    else:
        c = color
        opacity = 1.
    hex_color = '0x%02x%02x%02x' % (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
    return hex_color, opacity

def renderPoint(vis: meshcat.Visualizer, point: np.ndarray, point_name: str,
                color=np.ones(4), radius_point=0.001):
    hex_color, opacity = rgbToHex(color)
    vis[point_name].set_object(g.Sphere(radius_point), g.MeshLambertMaterial(color=hex_color, opacity=opacity))
    vis[point_name].set_transform(tf.translation_matrix(point))

def draw_pc(scene_rmc, vis, depth, K, se3_floor):
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
        o = Object.create_sphere(0.003, color=np.array([214, 39, 40])/255)
        scene_rmc.add_object(o)
        o.pose = se3.homogeneous
        renderPoint(vis, Xy[i], f"point_{i}", color=np.array([100, 22, 22, 255]) / 255, radius_point=3e-3)

    loader = hppfcl.MeshLoader()
    path = str(FLOOR_MESH_PATH)
    mesh: hppfcl.BVHModelBase = loader.load(path, scale=np.array([0.00075]*3))
    shape = pin.visualize.meshcat_visualizer.loadMesh(mesh)
    draw_shape(vis, shape, "plane", se3_floor, np.array([22, 145, 22, 255]) / 255)

def draw_pc_and_objects(ds_name: str):
    with open(FLOOR_POSES_PATH / "hope_bop_floor_poses_1mm_res_optimized.json", "r") as f:
        floor_se3s = json.load(f)
    rigid_objects = load_meshes(MESHES_PATH / ds_name, convex=False)
    vis = meshcat.Visualizer()
    scene_rmc = Scene()
    rigid_objects_rmc, o_floor = load_shapes_robomeshcat(ds_name, rigid_objects)
    for scene in floor_se3s:
        with open(DATASETS_PATH / ds_name / f"{int(scene):06d}" / "scene_gt.json", "r") as f:
            gt = json.load(f)
        for im in floor_se3s[scene]:
            print(scene, im)
            with open(DATASETS_PATH / ds_name / f"{int(scene):06d}" / "scene_camera.json", "r") as f:
                cam_json = json.load(f)
                K = np.array(cam_json[im]["cam_K"]).reshape(3,3)
            gt_poses = {}
            for obj in gt[im]:
                gt_poses[obj["obj_id"]] = pin.SE3(np.array(obj["cam_R_m2c"]).reshape(3,3), np.array(obj["cam_t_m2c"])/1000)
            depth = np.array(Image.open(DATASETS_PATH / ds_name / f"{int(scene):06d}" / "depth" / f"{int(im):06d}.png"))
            for i in range(len(rigid_objects)):
                if (DATASETS_PATH / ds_name / f"{int(scene):06d}" / "mask" / f"{int(im):06d}_{i:06d}.png").is_file():
                    mask = Image.open(DATASETS_PATH / ds_name / f"{int(scene):06d}" / "mask" / f"{int(im):06d}_{i:06d}.png")
                    mask = np.array(mask) !=  0
                    mask = ndimage.binary_dilation(mask, iterations=10)
                    depth[mask] = 0
            se3_floor = floor_se3s[scene][im]
            se3_floor = None if se3_floor is None else pin.SE3(np.array(se3_floor["R"]), np.array(se3_floor["t"]))
            if se3_floor is not None:
                o_floor.pose = se3_floor.homogeneous
                draw_pc(scene_rmc, vis, depth, K, se3_floor)
                for label in gt_poses:
                    scene_rmc.add_object(rigid_objects_rmc[str(label)])
                    rigid_objects_rmc[str(label)].pose = gt_poses[label].homogeneous
                    draw_shape(vis, rigid_objects[str(label)], f"{label}", gt_poses[label], np.array([31, 119, 180, 125]) / 255)
                print()
            else:
                print("No floor")
            input("Press enter to continue")