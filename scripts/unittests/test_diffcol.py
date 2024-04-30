import pydiffcol
import pinocchio as pin
import numpy as np
from tqdm import tqdm
from src.scene import SelectStrategyConfig
from pydiffcol.utils import (
    select_strategy
)
from eval.eval_utils import load_meshes, load_meshes_decomp
from config import MESHES_PATH, MESHES_DECOMP_PATH

meshes_ds_name = "tless"

args = SelectStrategyConfig(1e-2, 100, 1, "finite_differences")
col_req, col_req_diff = select_strategy(args)

rigid_objects = load_meshes(MESHES_PATH / meshes_ds_name)
rigid_objects_decomp = load_meshes_decomp(MESHES_DECOMP_PATH / meshes_ds_name)

labels = list(rigid_objects.keys())

col_res = pydiffcol.DistanceResult()
col_res_diff = pydiffcol.DerivativeResult()

convex_1_decomp_num = True
convex_2_decomp_num = True

for i in tqdm(range(1000000)):
    M1 = pin.SE3.Random()
    M2 = pin.SE3.Random()
    M1.translation = np.random.normal(0, 0.01, 3)
    M2.translation = np.random.normal(0, 0.01, 3)

    label_1 = np.random.choice(labels)
    label_2 = np.random.choice(labels)
    
    if np.random.rand() < 0.5:
        label_1_decomp_num = None
        convex_1 = rigid_objects[label_1]
    else:
        label_1_decomp_num = np.random.randint(len(rigid_objects_decomp[label_1]))
        convex_1 = rigid_objects_decomp[label_1][label_1_decomp_num]

    if np.random.rand() < 0.5:
        label_2_decomp_num = None
        convex_2 = rigid_objects[label_2]
    else:
        label_2_decomp_num = np.random.randint(len(rigid_objects_decomp[label_2]))
        convex_2 = rigid_objects_decomp[label_2][label_2_decomp_num]

    #print(f"Testing {label_1} {'hull' if not label_1_decomp_num else label_1_decomp_num} and {label_2} {'hull' if not label_2_decomp_num else label_2_decomp_num}")
    #print(f"Testing {M1} and {M2}")

    pydiffcol.distance(convex_1, M1, convex_2, M2, col_req, col_res)
    pydiffcol.distance_derivatives(convex_1, M1, convex_2, M2, col_req, col_res, col_req_diff, col_res_diff)