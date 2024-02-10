import pickle
from typing import List, Dict, Tuple
import numpy as np
import pinocchio as pin
import meshcat
import hppfcl
import pydiffcol
from dataclasses import dataclass

from pydiffcol.utils import (
    constructPolyhedralEllipsoid,
    constructDiamond, 
)
from pydiffcol.utils_render import draw_shape, draw_witness_points, RED, GREEN, BLUE

from spatial import normalize_se3


def create_shape(shape_type: str, num_subdiv=1, path_mesh=None):
    if shape_type == "diamond":
        shape = constructDiamond(0.5)
        print(f"Number vertices shape1: {shape.num_points}")
    elif shape_type == "poly_ellipsoid":
        r = np.array([0.1, 0.2, 0.3])
        shape = constructPolyhedralEllipsoid(r, num_subdiv)
        print(f"Number vertices shape1: {shape.num_points}")
    elif shape_type == "ellipsoid":
        r = np.array([0.1, 0.2, 0.3])
        shape = hppfcl.Ellipsoid(r)
    elif shape_type == "sphere":
        r = 0.5
        shape = hppfcl.Sphere(r)
    elif shape_type == "poly_sphere":
        r = np.array([0.1, 0.1, 0.1])
        shape = constructPolyhedralEllipsoid(r, num_subdiv)
        print(f"Number vertices shape1: {shape.num_points}")
    elif shape_type == "mesh":
        loader = hppfcl.MeshLoader()
        mesh: hppfcl.BVHModelBase = loader.load(path_mesh)
        mesh.buildCSelectStrategyConfigonvexHull(True, "Qt")
        shape = mesh.convex
    else:
        raise NotImplementedError
    return shape


def draw_scene(vis: meshcat.Visualizer,
               shape_lst: List[hppfcl.ShapeBase],
               stat_shape_lst: List[hppfcl.ShapeBase],
               wMo_lst: List[pin.SE3],
               wMs_lst: List[pin.SE3],
               col_res_pairs: Dict[Tuple[int,int], pydiffcol.DistanceResult],
               col_res_pairs_stat: Dict[Tuple[int,int], pydiffcol.DistanceResult],
               render_faces: bool = False,
               radius_points = 3e-3):
    
    in_collision_obj = {}
    for (id1, id2), col_res in col_res_pairs.items():
        col = col_res.dist < 0.0
        if id1 not in in_collision_obj:
            in_collision_obj[id1] = col
        if id2 not in in_collision_obj:
            in_collision_obj[id2] = col

        in_collision_obj[id1] = in_collision_obj[id1] or col
        in_collision_obj[id2] = in_collision_obj[id2] or col

        # M1, M2 = wMo_lst[id1], wMo_lst[id2]
        # draw_witness_points(vis, M1 * col_res.w1, M2 * col_res.w2, radius_points=radius_points)
    in_collision_stat = {}
    for (id_obj, id_stat), col_res in col_res_pairs_stat.items():
        col = col_res.dist < 0.0
        if id_stat not in in_collision_stat:
            in_collision_stat[id_stat] = col

        in_collision_obj[id_obj] = in_collision_obj[id_obj] or col
        in_collision_stat[id_stat] = in_collision_stat[id_stat] or col

    for i, (shape, M) in enumerate(zip(shape_lst, wMo_lst)):
        c = BLUE if in_collision_obj[i] else GREEN
        # TODO: bottlebeck, might be possible to speed up by calling once pydiffcol.utils_render.loadCVX 
        draw_shape(vis, shape, f"shape{i}", M, color=c, render_faces=render_faces)
    for i, (shape, M) in enumerate(zip(stat_shape_lst, wMs_lst)):
        c = BLUE if in_collision_stat[i] else GREEN
        draw_shape(vis, shape, f"stat_shape{i}", M, color=c, render_faces=render_faces)


def get_permutation_indices(N):
    """
    Generate combinations collision pairs.

    >>> get_permutation_indices(4)
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    """
    permutations = []
    for i in range(N):
        for j in range(i,N):
            if i != j:
                permutations.append((i,j))
    return permutations


def poses_equilateral(d):
    h = np.sqrt(3) * d /2

    wMo_lst = [
        pin.XYZQUATToSE3([0,0,0, 0,0,0,1]),
        pin.XYZQUATToSE3([d,0,0, 0,0,0,1]),
        pin.XYZQUATToSE3([d/2,h,0, 0,0,0,1]),
    ]
    return wMo_lst


def poses_tetraedron(d):
    s2 = np.sqrt(2)
    t_arr = d/2*np.array([
         1,  0, -1/s2,        
        -1,  0, -1/s2,
         0,  1,  1/s2,
         0, -1,  1/s2,
    ]).reshape((4,3))
    poses = [np.concatenate([t, [0,0,0,1]]) for t in t_arr]
    wMo_lst = [pin.XYZQUATToSE3(pose) for pose in poses]
    return wMo_lst


def read_poses_pickle(path: str):
    with open(path, 'rb') as f:
        scene_bproc = pickle.load(f)
    wMo_lst = [normalize_se3(pin.SE3(M)) for M in scene_bproc['wMo_lst']]
    wMc = normalize_se3(pin.SE3(scene_bproc['wMc']))
    return wMo_lst, wMc


@dataclass
class SelectStrategyConfig:
    """"""
    noise: float = 1e-3
    num_samples: int = 10
    max_neighbors_search_level: float = 1
    strategy: str = "first_order_gaussian"


class DiffColScene:
    """
    TODO: 
    - analytical shapes from hppfcl
    - broadphase 

    """
    def __init__(self, obj_paths: List[str], stat_obj_paths: List[str] = [], wMs_lst: List[pin.SE3] = []) -> None:

        assert len(stat_obj_paths) == len(wMs_lst)

        self.col_res_pairs = {}
        self.col_res_diff_pairs = {}
        self.col_res_pairs_stat = {}
        self.shapes = []
        self.stat_shapes = []
        self.mesh_loader = hppfcl.MeshLoader()

        for obj_path in obj_paths:
            self.shapes.append(self.create_mesh(obj_path))
        
        self.wMs_lst = wMs_lst
        for stat_obj_path in stat_obj_paths:
            self.stat_shapes.append(self.create_mesh(stat_obj_path))


    def compute_diffcol(self, wMo_lst: List[pin.SE3], col_req, col_req_diff, diffcol=True):
        # Compute col and diffcol for all pairs
        N = len(wMo_lst)
        index_pairs = get_permutation_indices(N)
        grad = np.zeros(6*N)
        cost_c = 0.0

        for i1, i2 in index_pairs:
            shape1, shape2 = self.shapes[i1], self.shapes[i2] 
            M1, M2 = wMo_lst[i1], wMo_lst[i2] 
            col_res = pydiffcol.DistanceResult()
            col_res_diff = pydiffcol.DerivativeResult()
            # TODO: implement broadphase
            _ = pydiffcol.distance(shape1, M1, shape2, M2, col_req, col_res)
            if diffcol and col_res.dist < 0:
                ############################################################################### FOR DECOMPOSITION distance derivatives convex dec
                pydiffcol.distance_derivatives(shape1, M1, shape2, M2, col_req, col_res, col_req_diff, col_res_diff)

            # include max(-phi(M1,M2), 0) gradient blocks
            if col_res.dist < 0 :
                cost_c += -col_res.dist
                grad[6*i1:6*i1+6] += -col_res_diff.ddist_dM1
                grad[6*i2:6*i2+6] += -col_res_diff.ddist_dM2

            self.col_res_pairs[(i1, i2)] = col_res
            self.col_res_diff_pairs[(i1, i2)] = col_res_diff

        return cost_c, grad
    

    def compute_diffcol_static(self, wMo_lst: List[pin.SE3], col_req, col_req_diff, diffcol=True):
        # Compute col and diffcol for all objects and floor
        N = len(wMo_lst)
        M = len(self.wMs_lst)
        grad = np.zeros(6*N)
        cost_c = 0.0

        for i in range(N):
            for j in range(M):
                shape_obj = self.shapes[i]
                shape_stat = self.stat_shapes[j]
                wMo = wMo_lst[i]
                wMs = self.wMs_lst[j]
                col_res = pydiffcol.DistanceResult()
                col_res_diff = pydiffcol.DerivativeResult()
                _ = pydiffcol.distance(shape_obj, wMo, shape_stat, wMs, col_req, col_res)
                if diffcol and col_res.dist < 0:
                    pydiffcol.distance_derivatives(shape_obj, wMo, shape_stat, wMs, col_req, col_res, col_req_diff, col_res_diff)
                if col_res.dist < 0:
                    cost_c += -col_res.dist
                    grad[6*i:6*i+6] += -col_res_diff.ddist_dM1

                self.col_res_pairs_stat[(i, j)] = col_res

        return cost_c, grad


    def create_mesh(self, obj_path: str):
        mesh: hppfcl.BVHModelBase = self.mesh_loader.load(obj_path)
        mesh.buildConvexHull(True, "Qt")
        return mesh.convex
