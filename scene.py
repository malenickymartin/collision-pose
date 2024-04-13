import pickle
from typing import List, Dict, Tuple, Union
import numpy as np
import pinocchio as pin
import meshcat
import hppfcl
import pydiffcol
from pathlib import Path
from dataclasses import dataclass

from pydiffcol.utils import (
    constructPolyhedralEllipsoid,
    constructDiamond, 
)

from spatial import normalize_se3
from eval.eval_utils import draw_shape, BLUE, GREEN


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
               col_res_pairs_stat: Dict[Tuple[int,int], pydiffcol.DistanceResult]
               ):
    
    in_collision_obj = {i:False for i in range(len(shape_lst))}
    for (id1, id2), col_dist in col_res_pairs.items():
        col = col_dist > 0.0
        in_collision_obj[id1] = in_collision_obj[id1] or col
        in_collision_obj[id2] = in_collision_obj[id2] or col

    in_collision_stat = {i:False for i in range(len(stat_shape_lst))}
    for (id_obj, id_stat), col_dist in col_res_pairs_stat.items():
        col = col_dist > 0.0
        if id_stat not in in_collision_stat:
            in_collision_stat[id_stat] = col

        in_collision_obj[id_obj] = in_collision_obj[id_obj] or col
        in_collision_stat[id_stat] = in_collision_stat[id_stat] or col

    for i, (shape, M) in enumerate(zip(shape_lst, wMo_lst)):
        c = BLUE if in_collision_obj[i] else GREEN
        draw_shape(vis, shape, f"shape{i}", M, color=c)
    for i, (shape, M) in enumerate(zip(stat_shape_lst, wMs_lst)):
        c = BLUE if in_collision_stat[i] else GREEN
        draw_shape(vis, shape, f"stat_shape{i}", M, color=c)


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
    Class for handling collision detection, distance computation, derivatives and gravity.
    
    Inputs:
    - obj_paths: list of paths to object meshes or list of pre-loaded hppfcl.Convex
    - stat_paths: list of paths to static object meshes or list of pre-loaded hppfcl.Convex, object in the first position is used as reference for gravity
    - wMs_lst: list of poses of static objects, object in the first position is used as reference for gravity
    - obj_decomp_paths: list of paths to directories with convex decompositions of objects or list of lists of pre-loaded hppfcl.Convex
    - stat_decomp_paths: list of paths to directories with convex decompositions of static objects or list of lists of pre-loaded hppfcl.Convex
    - scale: scale with which to load meshes
    - pre_loaded_meshes: bool, if True, obj_paths, stat_paths, obj_decomp_paths, stat_decomp_paths are lists of pre-loaded hppfcl.Convex

    ! Gravity is calculated towards to the first (index 0) static object. !

    """
    def __init__(
            self, obj_paths: List[Union[str, hppfcl.Convex]], stat_paths: List[Union[str, hppfcl.Convex]] = [],
            wMs_lst: List[pin.SE3] = [], obj_decomp_paths: List[Union[str, List[hppfcl.Convex]]] = [],
            stat_decomp_paths: List[Union[str, List[hppfcl.Convex]]] = [], scale: float = 1.0,
            pre_loaded_meshes: bool = False
            ) -> None:

        assert len(stat_paths) == len(wMs_lst)
        self.wMs_lst = wMs_lst

        self.col_res_pairs = {}
        self.col_res_diff_pairs = {}
        self.col_res_pairs_stat = {}

        self.shapes_convex = []
        self.shapes_decomp = []
        self.statics_convex = []
        self.statics_decomp = []

        self.mesh_loader = hppfcl.MeshLoader()
        self.scale = scale

        if pre_loaded_meshes:
            self.shapes_convex = obj_paths
            self.statics_convex = stat_paths
            self.shapes_decomp = obj_decomp_paths
            self.statics_decomp = stat_decomp_paths
        else:
            print("Loading meshes...")
            for path in obj_paths:
                self.shapes_convex.append(self.create_mesh(path))

            for path in obj_decomp_paths:
                self.shapes_decomp.append(self.create_decomposed_mesh(path))
            
            for path in stat_paths:
                self.statics_convex.append(self.create_mesh(path))
            
            for path in stat_decomp_paths:
                self.statics_decomp.append(self.create_decomposed_mesh(path))
            print("Meshes loaded.")

    def compute_gravity(self, wMo_lst: List[pin.SE3], grad_c_stat: np.ndarray, grad_c_obj: np.ndarray) -> np.ndarray:
        """
        Compute gravity vector for all objects.

        Inputs:
        - wMo_lst: list of poses of objects
        - wMc: pose of camera
        - g: np.ndarray of shape (3,)

        Returns:
        - f_lst: list of forces
        """
        g_s = np.array([0, 0, 9.81])
        g_c = self.wMs_lst[0].rotation @ g_s
        N = len(wMo_lst)
        grad = np.zeros(6*N)
        g_o_rot = np.zeros(3)
        for i in range(N):
            if np.allclose(grad_c_stat[6*i:6*(i+1)], 0, atol=1e-6) and np.allclose(grad_c_obj[6*i:6*(i+1)], 0, atol=1e-6):
                g_o_trans = wMo_lst[i].rotation.T @ g_c
                grad[6*i:6*(i+1)] = np.concatenate([g_o_trans, g_o_rot])
        return grad

    def compute_diffcol(self, wMo_lst: List[pin.SE3], 
                        col_req: hppfcl.DistanceRequest, col_req_diff: pydiffcol.DerivativeRequest,
                        coll_exp_scale: int = 0, diffcol: bool = True) -> Tuple[float, np.ndarray]:
        """
        Compute diffcol for all objects.

        Inputs:
        - wMo_lst: list of poses of objects
        - col_req, col_req_diff: hppfcl.DistanceRequest, pydiffcol.DerivativeRequest
        - coll_exp_scale: int
        - diffcol: bool

        Returns:
        - cost_c: float
        - grad: np.ndarray of shape (6*N,)
        """
        N = len(wMo_lst)
        index_pairs = get_permutation_indices(N)
        grad = np.zeros(6*N)
        cost_c = 0.0

        for i1, i2 in index_pairs:
            shape1, shape2 = self.shapes_convex[i1], self.shapes_convex[i2] 
            M1, M2 = normalize_se3(wMo_lst[i1]), normalize_se3(wMo_lst[i2])
            if len(self.shapes_decomp) > 0:
                shape1_decomp, shape2_decomp = self.shapes_decomp[i1], self.shapes_decomp[i2]
                max_coll_dist, grad_1, grad_2 = self.compute_diffcol_decomp(shape1, shape1_decomp, M1,
                                                                            shape2, shape2_decomp, M2, 
                                                                            col_req, col_req_diff, diffcol)
            else:
                max_coll_dist, grad_1, grad_2 = self.compute_diffcol_convex(shape1, M1,
                                                                            shape2, M2,
                                                                            col_req, col_req_diff, diffcol)
                
            if max_coll_dist > 0:
                cost_c += max_coll_dist
                grad[6*i1:6*i1+6] += np.exp(coll_exp_scale*max_coll_dist)*grad_1
                grad[6*i2:6*i2+6] += np.exp(coll_exp_scale*max_coll_dist)*grad_2

            self.col_res_pairs[(i1, i2)] = max_coll_dist

        return cost_c, grad


    def compute_diffcol_static(self, wMo_lst: List[pin.SE3],
                               col_req: hppfcl.DistanceRequest, col_req_diff: pydiffcol.DerivativeRequest,
                               coll_exp_scale: int = 0, diffcol=True):
        """
        Compute diffcol for all objects with static objects.

        Inputs:
        - wMo_lst: list of poses of objects
        - col_req, col_req_diff: hppfcl.DistanceRequest, pydiffcol.DerivativeRequest
        - coll_exp_scale: int
        - diffcol: bool

        Returns:
        - cost_c: float
        - grad: np.array of shape (6*N,)
        """
        N = len(wMo_lst)
        M = len(self.wMs_lst)
        grad = np.zeros(6*N)
        cost_c = 0.0

        for i1 in range(N):
            for i2 in range(M):
                shape_convex, wMo = self.shapes_convex[i1], normalize_se3(wMo_lst[i1])
                static_convex, wMs = self.statics_convex[i2], normalize_se3(self.wMs_lst[i2])

                if len(self.shapes_decomp) == 0 and len(self.statics_decomp) == 0: # both shapes and statics are convex
                    max_coll_dist, grad_1, _ = self.compute_diffcol_convex(shape_convex, wMo,
                                                                           static_convex, wMs,
                                                                           col_req, col_req_diff, diffcol)
                else:
                    if len(self.shapes_decomp) > 0 and len(self.statics_decomp) > 0: # both shapes and statics are decomposed
                        shape_decomp, static_decomp = self.shapes_decomp[i1], self.statics_decomp[i2]
                    elif len(self.shapes_decomp) > 0 and len(self.statics_decomp) == 0: # only shapes are decomposed
                        shape_decomp, static_decomp = self.shapes_decomp[i1], [self.statics_convex[i2]]
                    else: # only statics are decomposed
                        shape_decomp, static_decomp = [self.shapes_convex[i1]], self.statics_decomp[i2]
                    max_coll_dist, grad_1, _ = self.compute_diffcol_decomp(shape_convex, shape_decomp, wMo,
                                                    static_convex, static_decomp, wMs, 
                                                    col_req, col_req_diff, diffcol)
                    
                if max_coll_dist > 0: # if there is a collision between object and static object
                    cost_c += max_coll_dist
                    grad[6*i1:6*i1+6] += np.exp(coll_exp_scale*max_coll_dist)*grad_1
                self.col_res_pairs_stat[(i1, i2)] = max_coll_dist

        return cost_c, grad
    
    def compute_diffcol_convex(
            self, convex_1: hppfcl.Convex, M1: pin.SE3, convex_2: hppfcl.Convex, M2: pin.SE3,
            col_req: hppfcl.DistanceRequest, col_req_diff: pydiffcol.DerivativeRequest, diffcol=True
            ) -> Tuple[float, np.ndarray, np.ndarray, pydiffcol.DistanceResult, pydiffcol.DerivativeResult]:
        """
        Compute diffcol using only convex hulls.

        Inputs:
        - convex_1, convex_2: hppfcl.Convex
        - M1, M2: pin.SE3
        - col_req, col_req_diff: hppfcl.DistanceRequest, pydiffcol.DerivativeRequest
        - diffcol: bool

        Returns:
        - max_coll_dist: float
        - grad_1, grad_2: np.ndarray of shape (6,)
        """
        col_res = pydiffcol.DistanceResult()
        col_res_diff = pydiffcol.DerivativeResult()

        grad_1 = np.zeros(6)
        grad_2 = np.zeros(6)
        max_coll_dist = 0

        _ = pydiffcol.distance(convex_1, M1, convex_2, M2, col_req, col_res)
        if col_res.dist >= 0:
            # no collision between convex hulls
            return max_coll_dist, grad_1, grad_2
        
        if diffcol:
            pydiffcol.distance_derivatives(convex_1, M1, convex_2, M2, col_req, col_res, col_req_diff, col_res_diff)
            grad_1 = -col_res_diff.ddist_dM1
            grad_2 = -col_res_diff.ddist_dM2
        max_coll_dist = -col_res.dist
    
        return max_coll_dist, grad_1, grad_2

    def compute_diffcol_decomp(
            self, convex_1: hppfcl.Convex, decomp_1: List[hppfcl.Convex], M1: pin.SE3,
            convex_2: hppfcl.Convex, decomp_2: List[hppfcl.Convex], M2: List[pin.SE3],
            col_req: hppfcl.DistanceRequest, col_req_diff: pydiffcol.DerivativeRequest, diffcol: bool = True
            ) -> Tuple[float, np.ndarray, np.ndarray, pydiffcol.DistanceResult, pydiffcol.DerivativeResult]:
        """
        Compute diffcol using both convex hulls and convex decomposed shapes.

        Inputs:
        - convex_1, convex_2: hppfcl.Convex
        - decomp_1, decomp_2: list of hppfcl.Convex
        - M1, M2: pin.SE3
        - col_req, col_req_diff: hppfcl.DistanceRequest, pydiffcol.DerivativeRequest
        - diffcol: bool

        Returns:
        - max_coll_dist: float
        - grad_1, grad_2: np.array of shape (6,)
        """
        col_res = pydiffcol.DistanceResult()
        col_res_diff = pydiffcol.DerivativeResult()

        grad_1 = np.zeros(6)
        grad_2 = np.zeros(6)
        max_coll_dist = 0
        # TODO: implement broadphase
        _ = pydiffcol.distance(convex_1, M1, convex_2, M2, col_req, col_res)
        if col_res.dist >= 0:
            # no collision between convex hulls
            return max_coll_dist, grad_1, grad_2
        
        # check which parts of decomp_1 are in collision with convex_2
        decomp_1_in_coll = []
        for part_1 in decomp_1:
            _ = pydiffcol.distance(part_1, M1, convex_2, M2, col_req, col_res)
            if col_res.dist < 0:
                decomp_1_in_coll.append(True)
            else:
                decomp_1_in_coll.append(False)

        # check which parts of decomp_2 are in collision with convex_1
        decomp_2_in_coll = []
        for part_2 in decomp_2:
            _ = pydiffcol.distance(part_2, M2, convex_1, M1, col_req, col_res)
            if col_res.dist < 0:
                decomp_2_in_coll.append(True)
            else:
                decomp_2_in_coll.append(False)

        num_colls = 0
        for i, part_1 in enumerate(decomp_1):
            if decomp_1_in_coll[i]:
                for j, part_2 in enumerate(decomp_2):
                    if decomp_2_in_coll[j]:
                        _ = pydiffcol.distance(part_1, M1, part_2, M2, col_req, col_res)
                        if col_res.dist < 0 and diffcol:
                            pydiffcol.distance_derivatives(part_1, M1, part_2, M2, col_req, col_res, col_req_diff, col_res_diff)
                            grad_1 += -col_res_diff.ddist_dM1
                            grad_2 += -col_res_diff.ddist_dM2
                            num_colls += 1
                        if col_res.dist < max_coll_dist:
                            max_coll_dist = col_res.dist

        if num_colls > 0:
            grad_1 /= num_colls
            grad_2 /= num_colls
        max_coll_dist = -max_coll_dist
    
        return max_coll_dist, grad_1, grad_2


    def create_mesh(self, obj_path: str):
        """
        Loads mesh and creates convex hull.
        
        Inputs:
        - obj_path: str

        Returns:
        - hppfcl.Convex
        """
        mesh: hppfcl.BVHModelBase = self.mesh_loader.load(obj_path, scale=np.array(3*[self.scale]))
        mesh.buildConvexHull(True, "Qt")
        return mesh.convex

    
    def create_decomposed_mesh(self, dir_path: str):
        """
        Iterates through given directory with convex decompositions and creates a list of convex shapes.

        Inputs:
        - dir_path: str

        Returns:
        - list of hppfcl.Convex
        """
        meshes = []
        for path in Path(dir_path).iterdir():
            if path.suffix == ".ply" or path.suffix == ".obj":
                mesh: hppfcl.BVHModelBase = self.mesh_loader.load(str(path), scale=np.array(3*[self.scale]))
                mesh.buildConvexHull(True, "Qt")
                meshes.append(mesh.convex)
        return meshes
