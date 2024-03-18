from config import DATASETS_PATH, MESHES_PATH, FLOOR_MESH_PATH
from eval.eval_utils import get_floor_se3s, load_meshes
import numpy as np
import hppfcl

ds_name = "tless" # <=== Input
save_name = "tless_bop_floor_poses_1mm_res_dilation.json" # <=== Input

scenes_path = DATASETS_PATH / ds_name
path_meshes = MESHES_PATH / ds_name
rigid_objects = load_meshes(path_meshes)
loader = hppfcl.MeshLoader()
floor_path = str(FLOOR_MESH_PATH)
floor_hppfcl: hppfcl.BVHModelBase = loader.load(floor_path, scale = np.array([0.0005]*3))
floor_hppfcl.buildConvexHull(True, "Qt")
floor_mesh = floor_hppfcl.convex
get_floor_se3s(scenes_path, rigid_objects, floor_mesh, save_name)