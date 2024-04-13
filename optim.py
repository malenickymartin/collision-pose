from typing import List, Dict
import numpy as np
from numpy.linalg import norm
import pinocchio as pin
import copy


def change_Q_frame(var_xy_z_theta: List, M: pin.SE3) -> np.ndarray:

    t_c_o = M.translation
    rot_c_o = M.rotation

    var_xy, var_z, var_theta = var_xy_z_theta

    cov_trans_cp = np.diag([var_xy, var_xy, var_z])
    t_o_norm = t_c_o/np.linalg.norm(t_c_o)
    v = np.cross([0, 0, 1], t_o_norm)
    ang = np.arccos(np.dot([0, 0, 1], t_o_norm))
    v_norm = np.linalg.norm(v)
    if v_norm > 1e-6:
        v = v/v_norm
    rot_c_cp = pin.exp3(ang * v)
    cov_trans_c = rot_c_cp @ cov_trans_cp @ rot_c_cp.T  # cov[AZ] = A cov[Z] A^T
    
    cov_rot_c = rot_c_o @ np.diag([var_theta] * 3) @ rot_c_o.T

    cov_c = np.zeros((6, 6))
    cov_c[:3, :3] = cov_trans_c
    cov_c[3:6, 3:6] = cov_rot_c

    return cov_c

def std_to_Q_aligned(std_xy_z_theta: List[float], Mm: pin.SE3) -> np.ndarray:
    """
    Convert standard deviations to covariances aligned to to object.
    """
    std_xy, std_z, std_theta = std_xy_z_theta
    var_xy, var_z, var_theta = std_xy**2, std_z**2, std_theta**2
    Q_aligned = change_Q_frame([var_xy, var_z, var_theta], Mm)
    return Q_aligned


def cov_to_sqrt_prec(cov: np.ndarray) -> np.ndarray:
    """
    Convert covariance to the precision matrix.
    """
    # Inverse the covariance to get an precision matrix
    H = np.linalg.inv(cov)  
    # Compute the square root of the precision matrix as its Cholesky decomposition
    # Numpy uses H = L @ L.T convention, L -> lower triangular matrix
    L = np.linalg.cholesky(H)  
    return L


def error_se3(M: pin.SE3, Mm: pin.SE3, jac=False):
    """
    Happypose measurement distance residual and gradient.

    M: estimated pose
    Mm: measured pose
    """
    Mrel = Mm.inverse()*M
    e = pin.log(Mrel).vector
    if jac:
        J = pin.Jlog6(Mrel)
        return e, J
    else:
        return e


def error_r3_so3(M: pin.SE3, Mm: pin.SE3, jac=False):
    """
    Happypose measurement distance residual and gradient.

    M: estimated pose
    Mm: measured pose
    """
    et = M.translation - Mm.translation
    Rrel = Mm.rotation.T@M.rotation
    eo = pin.log3(Rrel)
    e = np.concatenate([et,eo])
    J = np.zeros((6,6))
    if jac:
        J[:3,:3] = M.rotation
        J[3:,3:] = pin.Jlog3(Rrel)
        return e, J
    else:
        return e


def perception_res_grad(M_lst: List[pin.SE3], Mm_lst: List[pin.SE3], L_lst: List[np.ndarray], error_fun=error_se3):
    """
    Compute residuals and gradients for all pose estimates|measurement pairs.

    For one pose, compute error e and jacobian matrix J:=de/dM.
    The perception cost function is by definition:
    cp(M) = 0.5 ||e(M)||^2_Q = 0.5 e.T Q^{-1} e
    
    with Q covariance of the measurement.
    The inverse of the covariance has been decomposed using Cholesky decomposition:
    Q^{-1} = L L.T
    so that we can write
    cp(M) = 0.5 (e.T L) (L.T e) = 0.5 ||L.T e||^2 

    The residuals are then defined as 
    r(M) = L.T e(M)

    and the gradient of cp as
    g := dcp/cM = dcp/dr dr/dM = r.T L.T J
    where J := de/dM is the error jacobian.


    Inputs:
    - M_lst: list of estimated poses
    - Mm_lst: list of measured poses
    - L_lst: list of Cholesky decompositions of the precision matrices

    Returns:
    - res: array of residuals
    - grad: array of gradients
    """
    assert len(M_lst) == len(Mm_lst)
    N = len(M_lst)
    grad = np.zeros(6*N)
    res = np.zeros(6*N)
    for i in range(N):
        L = L_lst[i]
        e, J = error_fun(M_lst[i], Mm_lst[i], True)
        r = L.T @ e
        g = r.T @ L.T @ J
        res[6*i:6*i+6], grad[6*i:6*i+6] = r, g
    
    return res, grad

def clip_grad(grad, thr_grad_t=100, thr_grad_R=100):
    """
    Clip the gradient to avoid large steps.
    """
    grad = copy.deepcopy(grad) # copy the gradient to avoid modifying the original
    grad = grad.reshape((-1,2,3)) # 2x3 matrix for each pose
    grad_norm = norm(grad, axis=-1) # norm of each 2x3 matrix
    mask = grad_norm > np.array([thr_grad_t, thr_grad_R]) # mask for large gradients
    if np.any(mask): # if there are large gradients
        with np.errstate(divide='ignore', invalid='ignore'): # ignore division by zero
            grad_normed = grad/grad_norm.reshape(-1,2,1) # normalize gradients
        thrs = np.array([thr_grad_t, thr_grad_R]).reshape(1,2,1) # thresholds for large gradients
        grad[mask] = (thrs*grad_normed)[mask] # clip large gradients
    return grad.reshape(-1) # return clipped gradients


def rplus_se3(M, dm):
    """
    Right-plus operation for SE3.

    M: SE3 object
    dm: se3 tangent space "delta"
    """
    return M*pin.exp(dm)


def update_est(X: List[pin.SE3], dx: np.ndarray):
    """
    Update estimated poses.

    X: list of object pose object variables
    dx: update step as an array of se3 tangent space "deltas"
    """
    assert 6*len(X) == len(dx)
    return [rplus_se3(M,dx[6*i:6*i+6]) for i, M in enumerate(X)]
