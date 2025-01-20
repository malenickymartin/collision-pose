import pinocchio as pin
import hppfcl
import numpy as np
from time import time
from tqdm import tqdm

import pydiffcol
from pydiffcol.utils import select_strategy

from eval.eval_utils import load_meshes
from config import MESHES_PATH

def distance_derivative(shape_1: hppfcl.Convex, M1: pin.SE3,
                        shape_2: hppfcl.Convex, M2: pin.SE3,
                        col_req, EPS=0.005) -> np.ndarray:
    col_res_plus = hppfcl.DistanceResult()
    col_res_minus = hppfcl.DistanceResult()
    grad = np.zeros(6)
    for i in range(6):
        dM = np.zeros(6)
        dM[i] = EPS
        M1_tmp = M1*pin.exp(dM)
        col_res_plus.clear()
        hppfcl.distance(shape_1, M1_tmp, shape_2, M2, col_req, col_res_plus)
        M1_tmp = M1*pin.exp(-dM)
        col_res_minus.clear()
        hppfcl.distance(shape_1, M1_tmp, shape_2, M2, col_req, col_res_minus)
        grad[i] = (col_res_plus.min_distance - col_res_minus.min_distance) / (2*EPS)
    return grad

def distance_derivative_zero_order_gaussian(shape_1: hppfcl.Convex, M1: pin.SE3,
                        shape_2: hppfcl.Convex, M2: pin.SE3,
                        col_req, noise=0.005, samples=100) -> np.ndarray:
    col_res = hppfcl.DistanceResult()
    dist_0 = hppfcl.distance(shape_1, M1, shape_2, M2, col_req, col_res)
    grad = np.zeros(6)
    for _ in range(samples):
        dM = np.random.normal(0, 1, 6) * noise
        M1_tmp = M1*pin.exp(dM)
        col_res.clear()
        dist = hppfcl.distance(shape_1, M1_tmp, shape_2, M2, col_req, col_res)
        grad += ((dist - dist_0) * dM)/ noise
    return grad/samples

EPS = 1e-5
iters = 1000
pydiffcol.set_seed(1)
pin.seed(1)
np.random.seed(1)

col_req_hppfcl = hppfcl.DistanceRequest()
col_req_fd, col_req_diff_fd = select_strategy(type("SelectStrategyConfig", (object,),
                                        {"noise": EPS, "num_samples": 500, "max_neighbors_search_level": 2, "strategy": "finite_differences"})())
col_req_fog, col_req_diff_fog = select_strategy(type("SelectStrategyConfig", (object,),
                                        {"noise": EPS, "num_samples": 500, "max_neighbors_search_level": 2, "strategy": "first_order_gumbel"})())
col_req_zog, col_req_diff_zog = select_strategy(type("SelectStrategyConfig", (object,),
                                        {"noise": EPS, "num_samples": 500, "max_neighbors_search_level": 2, "strategy": "zero_order_gaussian"})())

col_res = pydiffcol.DistanceResult()
col_res_diff_fd = pydiffcol.DerivativeResult()
col_res_diff_fog = pydiffcol.DerivativeResult()
col_res_diff_zog = pydiffcol.DerivativeResult()

rigid_objects = load_meshes(MESHES_PATH / "ycbv") # dict where keys are labels of meshes and values are hpp-fcl convex hulls of the meshes
labels = list(rigid_objects.keys())

time_hppfcl = 0
time_pydiffcol_fd = 0
time_pydiffcol_fog = 0
time_pydiffcol_zog = 0
diff_grad = 0

for i in tqdm(range(iters)):
    M1 = pin.SE3.Random()
    M2 = pin.SE3.Random()
    M1.translation = np.random.normal(0, 0.05, 3)
    M2.translation = np.random.normal(0, 0.05, 3)

    label_1 = np.random.choice(labels)
    label_2 = np.random.choice(labels)
    
    convex_1 = rigid_objects[label_1]
    convex_2 = rigid_objects[label_2]


    pydiffcol.distance(convex_1, M1, convex_2, M2, col_req_fd, col_res)
    t0 = time()
    pydiffcol.distance_derivatives(convex_1, M1, convex_2, M2, col_req_fd, col_res, col_req_diff_fd, col_res_diff_fd)
    time_pydiffcol_fd += time() - t0

    pydiffcol.distance(convex_1, M1, convex_2, M2, col_req_fd, col_res)
    t0 = time()
    pydiffcol.distance_derivatives(convex_1, M1, convex_2, M2, col_req_fog, col_res, col_req_diff_fog, col_res_diff_fog)
    time_pydiffcol_fog += time() - t0

    t0 = time()
    grad_hppfcl = distance_derivative(convex_1, M1, convex_2, M2, col_req_hppfcl, EPS)
    time_hppfcl += time() - t0

    pydiffcol.distance(convex_1, M1, convex_2, M2, col_req_fd, col_res)
    t0 = time()
    pydiffcol.distance_derivatives(convex_1, M1, convex_2, M2, col_req_zog, col_res, col_req_diff_zog, col_res_diff_zog)
    time_pydiffcol_zog += time() - t0

    if not np.allclose(grad_hppfcl, col_res_diff_fd.ddist_dM1, atol=1e-3):
        diff_grad += 1

print(f"Avg. time pydiffcol fd: {time_pydiffcol_fd/iters}")
print(f"Avg. time pydiffcol fog: {time_pydiffcol_fog/iters}")
print(f"Avg. time my fd: {time_hppfcl/iters}")
#print(f"Ratio of different gradients: {diff_grad/iters}")