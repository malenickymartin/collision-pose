from typing import List, Dict
import numpy as np
from numpy.linalg import norm
import pinocchio as pin
import copy


def change_Q_frame(Q: np.ndarray, M: pin.SE3):

    t_o = M.translation
    R_o = M.rotation

    var_xy, var_z, var_angle = Q

    cov_trans_cam_aligned = np.diag([var_xy, var_xy, var_z])
    v = np.cross([0, 0, 1], t_o/np.linalg.norm(t_o))
    ang = np.arccos(np.dot([0, 0, 1], t_o/np.linalg.norm(t_o)))
    rot = pin.exp3(ang * v/np.linalg.norm(v))
    cov_trans_c = rot @ cov_trans_cam_aligned @ rot.T  # cov[AZ] = A cov[Z] A^T
    rot = R_o.T
    cov_trans_o = rot @ cov_trans_c @ rot.T  # cov[AZ] = A cov[Z] A^T

    cov_o = np.zeros((6, 6))
    cov_o[:3, :3] = cov_trans_o
    cov_o[3:6, 3:6] = np.diag([var_angle] * 3)

    return cov_o


def res_grad_se3(M: pin.SE3, Mm: pin.SE3, Q: np.ndarray):
    """
    Happypose measurement distance residual and gradient.

    M: estimated pose
    Mm: measured pose
    Q: covariance (3 values for xy, z and angles, respectively)

    Return:
    - residual(M,Mm) = T - Tm := Log6(Tm.inv()@T)
    and 
    - gradient = res * dres/dM
    """
    Mrel = Mm.inverse()*M
    res = pin.log(Mrel).vector
    Q_obj = change_Q_frame(Q, Mm)
    Q_inv = np.linalg.inv(Q_obj)
    Q_sqrt = np.linalg.cholesky(Q_inv)
    res = Q_sqrt @ res
    J = pin.Jlog6(Mrel)
    return res, res @ J


def perception_res_grad(M_lst: List[pin.SE3], Mm_lst: List[pin.SE3], Q: np.ndarray = np.eye(6)):
    """
    Compute residuals and gradients for all pose estimates|measurement pairs.

    Inputs:
    - M_lst: list of estimated poses
    - Mm_lst: list of measured poses
    - Q: measurement noise covariance

    Returns:
    - res: array of residuals
    - grad: array of gradients
    """
    assert len(M_lst) == len(Mm_lst)
    N = len(M_lst)
    grad = np.zeros(6*N)
    res = np.zeros(6*N)
    for i in range(N):
        res[6*i:6*i+6], grad[6*i:6*i+6] = res_grad_se3(M_lst[i], Mm_lst[i], Q)
    
    return res, grad

def clip_grad(grad, thr_grad_t=500, thr_grad_R=500):
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
