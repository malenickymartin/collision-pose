import pydiffcol
from pydiffcol.utils import select_strategy
from src.scene import SelectStrategyConfig
from eval.eval_utils import load_meshes
from config import MESHES_PATH
import pinocchio as pin
import hppfcl
import numpy as np

args = SelectStrategyConfig(0.005, 100, 1, "finite_differences")
col_req_diffcol, col_req_diff = select_strategy(args)
col_res_diffcol = pydiffcol.DistanceResult()

col_req_hppfcl = hppfcl.DistanceRequest()
col_res_hppfcl = hppfcl.DistanceResult()

rigid_objects = load_meshes(MESHES_PATH / "ycbv")
labels = list(rigid_objects.keys())

cnt_diff = 0
cnt_coll = 0

iters = 10000
for i in range(iters):
    M1 = pin.SE3.Random()
    M2 = pin.SE3.Random()
    M1.translation = np.random.normal(0, 0.05, 3)
    M2.translation = np.random.normal(0, 0.05, 3)

    label_1 = np.random.choice(labels)
    label_2 = np.random.choice(labels)
    
    convex_1 = rigid_objects[label_1]
    convex_2 = rigid_objects[label_2]

    col_res_hppfcl.clear()
    pydiffcol.distance(convex_1, M1, convex_2, M2, col_req_diffcol, col_res_diffcol)
    hppfcl.distance(convex_1, M1, convex_2, M2, col_req_hppfcl, col_res_hppfcl)
    print(f"Distance: {col_res_diffcol.min_distance < 0}; Difference: {abs(col_res_diffcol.min_distance - col_res_hppfcl.min_distance)}")
    if abs(col_res_hppfcl.min_distance - col_res_hppfcl.min_distance) > 1e-6:
        cnt_diff += 1
    if col_res_hppfcl.min_distance < 0:
        cnt_coll += 1
        

print(f"Ratio of collision: {cnt_coll/iters}")
print(f"Total number of differences: {cnt_diff}")