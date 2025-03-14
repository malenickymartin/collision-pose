import meshcat
import pinocchio as pin
import numpy as np
from typing import List, Dict, Tuple
import pydiffcol
import hppfcl

GREEN = np.array([110, 250, 90, 200]) / 255
RED = np.array([200, 11, 50, 100]) / 255
BLUE = np.array([90, 110, 250, 125]) / 255


def meshcat_material(r, g, b, a) -> meshcat.geometry.MeshPhongMaterial:
    """
    Create a meshcat material with the given RGBA values.
    """
    material = meshcat.geometry.MeshPhongMaterial()
    material.color = int(r * 255) * 256 ** 2 + int(g * 255) * 256 + \
        int(b * 255)
    material.opacity = a
    return material


def get_ellipsoid(cov, nstd=2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get the radii and rotation matrix of the ellipsoid.
    https://users.cs.utah.edu/~tch/CS4640/resources/A%20geometric%20interpretation%20of%20the%20covariance%20matrix.pdf
    Inputs:
        cov: covariance matrix
        nstd: number of standard deviations
    Returns:
        radii: radii of the ellipsoid
        eigvecs: rotation matrix of the ellipsoid
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    radii = np.sqrt(eigvals) * nstd
    return radii, eigvecs


def show_cov_ellipsoid(vis, mean, cov, ellipsoid_id=1, nstd=2):
    """
    Show the covariance ellipsoid in the visualizer.
    Inputs:
        vis: meshcat visualizer
        mean: mean of the distribution
        cov: covariance matrix
        ellipsoid_id: id of the ellipsoid
        nstd: number of standard deviations
    """
    radii, R_ellipsoid = get_ellipsoid(cov, nstd)
    T_ellipsoid = pin.SE3(R_ellipsoid, mean)
    ellipsoid_mesh = meshcat.geometry.Ellipsoid(radii)
    vis[f"ellipsoid_{ellipsoid_id}"].set_object(ellipsoid_mesh, meshcat_material(*RED))
    vis[f"ellipsoid_{ellipsoid_id}"].set_transform(T_ellipsoid.homogeneous)


def get_transform(T_: hppfcl.Transform3f) -> np.ndarray:
    """
    Convert the transform to a 4x4 matrix.
    """
    T = np.eye(4)
    if isinstance(T_, hppfcl.Transform3f):
        T[:3, :3] = T_.getRotation()
        T[:3, 3] = T_.getTranslation()
    elif isinstance(T_, pin.SE3):
        T[:3, :3] = T_.rotation
        T[:3, 3] = T_.translation
    else:
        raise NotADirectoryError
    return T


def draw_shape(vis: meshcat.Visualizer, mesh: meshcat.geometry.TriangularMeshGeometry, name: str, M: pin.SE3, color: np.ndarray):
    """
    Draws the shape in the visualizer.
    Inputs:
        vis: meshcat visualizer
        mesh: meshcat object
        name: name of the object
        M: pose of the object (pinocchio.SE3)
        color: color of the object
    """
    vis[name].set_object(mesh, meshcat_material(*color))
    T = get_transform(M)
    vis[name].set_transform(T)


def draw_scene(vis: meshcat.Visualizer,
               shape_lst: List[hppfcl.ShapeBase],
               stat_shape_lst: List[hppfcl.ShapeBase],
               wMo_lst: List[pin.SE3],
               wMs_lst: List[pin.SE3],
               col_res_pairs: Dict[Tuple[int,int], pydiffcol.DistanceResult],
               col_res_pairs_stat: Dict[Tuple[int,int], pydiffcol.DistanceResult]
               ):
    """
    Draw the scene in the visualizer.
    Inputs:
        vis: meshcat visualizer
        shape_lst: list of shapes
        stat_shape_lst: list of static shapes
        wMo_lst: list of poses of the shapes
        wMs_lst: list of poses of the static shapes
        col_res_pairs: collision results between shapes
        col_res_pairs_stat: collision results between shapes and static shapes
    """    
    in_collision_obj = {i:False for i in range(len(shape_lst))}
    for (id1, id2), col_dist in col_res_pairs.items():
        col = col_dist > 1e-4
        in_collision_obj[id1] = in_collision_obj[id1] or col
        in_collision_obj[id2] = in_collision_obj[id2] or col

    in_collision_stat = {i:False for i in range(len(stat_shape_lst))}
    for (id_obj, id_stat), col_dist in col_res_pairs_stat.items():
        col = col_dist > 1e-4
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