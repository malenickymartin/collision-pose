from typing import List, Tuple, Union
import numpy as np
import pinocchio as pin
import hppfcl
import pydiffcol
from pathlib import Path
from dataclasses import dataclass

from src.optim_tools import normalize_se3


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


@dataclass
class SelectStrategyConfig:
    """"""
    noise: float = 1
    num_samples: int = 100
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

    def compute_gravity(self, wMo_lst: List[pin.SE3], 
                        col_req: hppfcl.DistanceRequest, col_req_diff: pydiffcol.DerivativeRequest,
                        num_colls: np.ndarray
                        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute gravity gradient and cost for all objects.

        Inputs:
        - wMo_lst: list of poses of objects
        - col_req, col_req_diff: hppfcl.DistanceRequest, pydiffcol.DerivativeRequest
        - cost_c_obj: np.ndarray of shape (N,)
        - cost_c_stat: np.ndarray of shape (N,)

        Returns:
        - cost_g: float of shape (N,)
        - grad: np.ndarray of shape (6*N,)
        """
        N = len(wMo_lst)
        grad = np.zeros(6*N)
        cost_g = np.zeros(N)

        floor_convex, wMs = self.statics_convex[0], normalize_se3(self.wMs_lst[0])
        if len(self.statics_decomp) > 0:
            floor_decomp = self.statics_decomp[0]
        else:
            floor_decomp = [floor_convex]

        for i in range(N):
            if num_colls[i] > 0:
                continue
            object_convex, wMo = self.shapes_convex[i], normalize_se3(wMo_lst[i])
            if len(self.shapes_decomp) > 0:
                object_decomp = self.shapes_decomp[i]
                grav_dist, grad_1 = self.compute_gravity_decomp(object_convex, object_decomp, wMo,
                                                                floor_convex, floor_decomp, wMs, 
                                                                col_req, col_req_diff)
            else:
                grav_dist, grad_1 = self.compute_gravity_convex(object_convex, wMo, 
                                                                floor_convex, wMs,
                                                                col_req, col_req_diff)

            if grav_dist > 0:    
                cost_g[i] += grav_dist
                grad[6*i:6*i+6] = grad_1
        
        return cost_g, grad
    

    def compute_diffcol(self, wMo_lst: List[pin.SE3], 
                        col_req: hppfcl.DistanceRequest, col_req_diff: pydiffcol.DerivativeRequest,
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute diffcol for all objects.

        Inputs:
        - wMo_lst: list of poses of objects
        - col_req, col_req_diff: hppfcl.DistanceRequest, pydiffcol.DerivativeRequest

        Returns:
        - cost_c: np.ndarray of shape (N,)
        - grad: np.ndarray of shape (6*N,)
        """
        N = len(wMo_lst)
        index_pairs = get_permutation_indices(N)
        grad = np.zeros(6*N)
        cost_c = np.zeros(N)
        num_colls = np.zeros(N)

        for i1, i2 in index_pairs:
            shape1, shape2 = self.shapes_convex[i1], self.shapes_convex[i2] 
            M1, M2 = normalize_se3(wMo_lst[i1]), normalize_se3(wMo_lst[i2])
            if len(self.shapes_decomp) > 0:
                shape1_decomp, shape2_decomp = self.shapes_decomp[i1], self.shapes_decomp[i2]
                coll_dist, grad_1, grad_2 = self.compute_diffcol_decomp(shape1, shape1_decomp, M1,
                                                                            shape2, shape2_decomp, M2, 
                                                                            col_req, col_req_diff)
            else:
                coll_dist, grad_1, grad_2 = self.compute_diffcol_convex(shape1, M1,
                                                                            shape2, M2,
                                                                            col_req, col_req_diff)
                
            if coll_dist > 0:
                num_colls[i1] += 1
                num_colls[i2] += 1
                cost_c[i1] += coll_dist
                cost_c[i2] += coll_dist
                grad[6*i1:6*i1+6] += grad_1
                grad[6*i2:6*i2+6] += grad_2
            self.col_res_pairs[(i1, i2)] = coll_dist

        return cost_c, grad, num_colls


    def compute_diffcol_static(self, wMo_lst: List[pin.SE3],
                               col_req: hppfcl.DistanceRequest, col_req_diff: pydiffcol.DerivativeRequest,
                               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute diffcol for all objects with static objects.

        Inputs:
        - wMo_lst: list of poses of objects
        - col_req, col_req_diff: hppfcl.DistanceRequest, pydiffcol.DerivativeRequest

        Returns:
        - cost_c: float
        - grad: np.array of shape (6*N,)
        """
        N = len(wMo_lst)
        M = len(self.wMs_lst)
        grad = np.zeros(6*N)
        cost_c = np.zeros(N)
        num_colls = np.zeros(N)

        for i1 in range(N):
            for i2 in range(M):
                shape_convex, wMo = self.shapes_convex[i1], normalize_se3(wMo_lst[i1])
                static_convex, wMs = self.statics_convex[i2], normalize_se3(self.wMs_lst[i2])

                if len(self.shapes_decomp) == 0 and len(self.statics_decomp) == 0: # both shapes and statics are convex
                    coll_dist, grad_1, _ = self.compute_diffcol_convex(shape_convex, wMo,
                                                                       static_convex, wMs,
                                                                       col_req, col_req_diff)
                else:
                    if len(self.shapes_decomp) > 0 and len(self.statics_decomp) > 0: # both shapes and statics are decomposed
                        shape_decomp, static_decomp = self.shapes_decomp[i1], self.statics_decomp[i2]
                    elif len(self.shapes_decomp) > 0 and len(self.statics_decomp) == 0: # only shapes are decomposed
                        shape_decomp, static_decomp = self.shapes_decomp[i1], [self.statics_convex[i2]]
                    else: # only statics are decomposed
                        shape_decomp, static_decomp = [self.shapes_convex[i1]], self.statics_decomp[i2]
                    coll_dist, grad_1, _ = self.compute_diffcol_decomp(shape_convex, shape_decomp, wMo,
                                                                       static_convex, static_decomp, wMs, 
                                                                       col_req, col_req_diff)
                    
                if coll_dist > 0: # if there is a collision between object and static object
                    num_colls[i1] += 1
                    cost_c[i1] += coll_dist
                    grad[6*i1:6*i1+6] += grad_1
                self.col_res_pairs_stat[(i1, i2)] = coll_dist

        return cost_c, grad, num_colls
    
    def compute_diffcol_convex(
            self, convex_1: hppfcl.Convex, M1: pin.SE3, convex_2: hppfcl.Convex, M2: pin.SE3,
            col_req: hppfcl.DistanceRequest, col_req_diff: pydiffcol.DerivativeRequest
            ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute diffcol using only convex hulls.

        Inputs:
        - convex_1, convex_2: hppfcl.Convex
        - M1, M2: pin.SE3
        - col_req, col_req_diff: hppfcl.DistanceRequest, pydiffcol.DerivativeRequest

        Returns:
        - coll_dist: float
        - grad_1, grad_2: np.ndarray of shape (6,)
        """
        col_res = pydiffcol.DistanceResult()
        col_res_diff = pydiffcol.DerivativeResult()

        grad_1 = np.zeros(6)
        grad_2 = np.zeros(6)
        coll_dist = 0

        _ = pydiffcol.distance(convex_1, M1, convex_2, M2, col_req, col_res)
        if col_res.dist >= 0:
            # no collision between convex hulls
            return coll_dist, grad_1, grad_2
        
        pydiffcol.distance_derivatives(convex_1, M1, convex_2, M2, col_req, col_res, col_req_diff, col_res_diff)
        grad_1 = -col_res_diff.ddist_dM1
        grad_2 = -col_res_diff.ddist_dM2
        coll_dist = -col_res.dist
    
        return coll_dist, grad_1, grad_2

    def compute_diffcol_decomp(
            self, convex_1: hppfcl.Convex, decomp_1: List[hppfcl.Convex], M1: pin.SE3,
            convex_2: hppfcl.Convex, decomp_2: List[hppfcl.Convex], M2: List[pin.SE3],
            col_req: hppfcl.DistanceRequest, col_req_diff: pydiffcol.DerivativeRequest
            ) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute diffcol using both convex hulls and convex decomposed shapes.

        Inputs:
        - convex_1, convex_2: hppfcl.Convex
        - decomp_1, decomp_2: list of hppfcl.Convex
        - M1, M2: pin.SE3
        - col_req, col_req_diff: hppfcl.DistanceRequest, pydiffcol.DerivativeRequest

        Returns:
        - coll_dist: float
        - grad_1, grad_2: np.array of shape (6,)
        """
        col_res = pydiffcol.DistanceResult()
        col_res_diff = pydiffcol.DerivativeResult()

        grad_1 = np.zeros(6)
        grad_2 = np.zeros(6)
        coll_dist = 0
        _ = pydiffcol.distance(convex_1, M1, convex_2, M2, col_req, col_res)
        if col_res.dist >= 0:
            # no collision between convex hulls
            return coll_dist, grad_1, grad_2
        
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

        if not (np.any(decomp_1_in_coll) and np.any(decomp_2_in_coll)):
            return coll_dist, grad_1, grad_2

        num_colls = 0
        for i, part_1 in enumerate(decomp_1):
            if decomp_1_in_coll[i]:
                for j, part_2 in enumerate(decomp_2):
                    if decomp_2_in_coll[j]:
                        _ = pydiffcol.distance(part_1, M1, part_2, M2, col_req, col_res)
                        if col_res.dist < 0:
                            coll_dist -= col_res.dist
                            num_colls += 1
                            pydiffcol.distance_derivatives(part_1, M1, part_2, M2, col_req, col_res, col_req_diff, col_res_diff)
                            grad_1 += -col_res_diff.ddist_dM1
                            grad_2 += -col_res_diff.ddist_dM2               

        if num_colls > 0:
            coll_dist = coll_dist/num_colls
            grad_1 /= num_colls
            grad_2 /= num_colls

        return coll_dist, grad_1, grad_2

    def compute_gravity_convex(
            self, object_convex: hppfcl.Convex, wMo: pin.SE3, floor_convex: hppfcl.Convex, wMs: pin.SE3,
            col_req: hppfcl.DistanceRequest, col_req_diff: pydiffcol.DerivativeRequest
            ) -> Tuple[float, np.ndarray]:
        """
        Compute gravity gradient and cost for one object.

        Inputs:
        - object_convex, floor_convex: hppfcl.Convex
        - wMo, wMs: pin.SE3
        - col_req, col_req_diff: hppfcl.DistanceRequest, pydiffcol.DerivativeRequest

        Returns:
        - grav_dist: float
        - grad: np.ndarray of shape (6,)
        """
        col_res = pydiffcol.DistanceResult()
        col_res_diff = pydiffcol.DerivativeResult()

        grad = np.zeros(6)
        grav_dist = 0

        _ = pydiffcol.distance(object_convex, wMo, floor_convex, wMs, col_req, col_res)
        if col_res.dist <= 0:
            # convex hulls on collision
            return grav_dist, grad
        
        pydiffcol.distance_derivatives(object_convex, wMo, floor_convex, wMs, col_req, col_res, col_req_diff, col_res_diff)
        grad = col_res_diff.ddist_dM1
        grav_dist = col_res.dist
    
        return grav_dist, grad

    def compute_gravity_decomp(
            self, object_convex: hppfcl.Convex, object_decomp: List[hppfcl.Convex], wMo: pin.SE3,
            floor_convex: hppfcl.Convex, floor_decomp: List[hppfcl.Convex], wMs: List[pin.SE3],
            col_req: hppfcl.DistanceRequest, col_req_diff: pydiffcol.DerivativeRequest
            ) -> Tuple[float, np.ndarray]:
        """
        Compute gravity gradient and cost for one object with decomposed shapes.

        Inputs:
        - object_convex, floor_convex: hppfcl.Convex
        - object_decomp, floor_decomp: list of hppfcl.Convex
        - wMo, wMs: pin.SE3
        - col_req, col_req_diff: hppfcl.DistanceRequest, pydiffcol.DerivativeRequest

        Returns:
        - grav_dist: float
        - grad: np.ndarray of shape (6,)
        """
        col_res = pydiffcol.DistanceResult()
        col_res_diff = pydiffcol.DerivativeResult()

        grad = np.zeros(6)
        grav_dist = 0
        _ = pydiffcol.distance(object_convex, wMo, floor_convex, wMs, col_req, col_res)
        if col_res.dist <= 0:
            # convex hulls in collision
            return grav_dist, grad

        num_grav = 0
        for i, part_1 in enumerate(floor_decomp):
                for j, part_2 in enumerate(object_decomp):
                    _ = pydiffcol.distance(part_1, wMs, part_2, wMo, col_req, col_res)
                    if col_res.dist >= 0:
                        grav_dist += col_res.dist
                        num_grav += 1
                        pydiffcol.distance_derivatives(part_1, wMs, part_2, wMo, col_req, col_res, col_req_diff, col_res_diff)
                        grad += col_res_diff.ddist_dM2

        if num_grav > 0:
            grav_dist = grav_dist/num_grav
            grad /= num_grav
    
        return grav_dist, grad

    def create_mesh(self, obj_path: str) -> hppfcl.Convex:
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

    def create_decomposed_mesh(self, dir_path: str) -> List[hppfcl.Convex]:
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
