from typing import List, Tuple
import numpy as np
from numpy.linalg import norm
import pinocchio as pin
import copy


def change_Q_frame(var_xy_z_theta: List, M: pin.SE3) -> np.ndarray:
    """
    Change the covariance matrix frame to the camera frame.
    Inputs:
        var_xy_z_theta: list of variances for x, y, z, theta
        M: pose of the object
    Returns:
        cov_c: covariance matrix in the camera frame
    """

    t_c_o = M.translation
    rot_c_o = M.rotation

    var_xy, var_z, var_theta = var_xy_z_theta
    cov_trans_cp = np.diag([var_xy, var_xy, var_z])

    t_c_o_norm = np.linalg.norm(t_c_o)
    if t_c_o_norm > 1e-6:
        t_c_o = t_c_o/t_c_o_norm
    v = np.cross([0, 0, 1], t_c_o)
    ang = np.arccos(np.dot([0, 0, 1], t_c_o))
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
    Inputs:
        std_xy_z_theta: list of standard deviations for x, y, z, theta
        Mm: measured pose
    Returns:
        Q_aligned: covariance matrix aligned to the object
    """
    std_xy, std_z, std_theta = std_xy_z_theta
    var_xy, var_z, var_theta = std_xy**2, std_z**2, std_theta**2
    Q_aligned = change_Q_frame([var_xy, var_z, var_theta], Mm)
    return Q_aligned


def cov_to_sqrt_prec(cov: np.ndarray) -> np.ndarray:
    """
    Convert covariance to the precision matrix.
    Inputs:
        cov: covariance matrix
    Returns:
        L: Cholesky decomposition of the precision matrix
    """
    # Inverse the covariance to get an precision matrix
    H = np.linalg.inv(cov)  
    # Compute the square root of the precision matrix as its Cholesky decomposition
    # Numpy uses H = L @ L.T convention, L -> lower triangular matrix
    L = np.linalg.cholesky(H)  
    return L


def error_se3(M: pin.SE3, Mm: pin.SE3, jac=False) -> np.ndarray:
    """
    Happypose measurement distance residual and gradient.
    Inputs:
        M: estimated pose
        Mm: measured pose
        jac: flag to compute the gradient
    Returns:
        residual vector or residual vector and gradient
    """
    Mrel = Mm.inverse()*M
    e = pin.log(Mrel).vector
    if jac:
        J = pin.Jlog6(Mrel)
        return e, J
    else:
        return e


def error_r3_so3(M: pin.SE3, Mm: pin.SE3, jac=False) -> np.ndarray:
    """
    Happypose measurement distance residual and gradient.
    Inputs:
        M: estimated pose
        Mm: measured pose
        jac: flag to compute the gradient
    Returns:
        residual vector or residual vector and gradient
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


def perception_res_grad(M_lst: List[pin.SE3], Mm_lst: List[pin.SE3], L_lst: List[np.ndarray], error_fun=error_se3) -> Tuple[np.ndarray, np.ndarray]:
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

def clip_grad(grad, thr_grad_t=100, thr_grad_R=100) -> np.ndarray:
    """
    Clip the gradient to avoid large steps.
    Inputs:
        grad: gradient
        thr_grad_t: threshold for translation part
        thr_grad_R: threshold for rotation part
    Returns:
        clipped gradient
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


def normalize_se3(M):
    """
    Normalize the quaternion part of the SE3 object and return the normalized SE3 object.
    Inputs:
        M: SE3 object
    Returns:
        SE3 object with normalized rotation part
    """
    pose = pin.SE3ToXYZQUAT(M)
    q_norm = np.linalg.norm(pose[3:])
    pose[3:] = pose[3:] / q_norm
    return pin.XYZQUATToSE3(pose)


def rplus_se3(M: pin.SE3, dm: np.ndarray) -> pin.SE3:
    """
    Right-plus operation for SE3.
    Inputs:
        M: SE3 object
        dm: se3 tangent space "delta"
    Returns:
        M*exp(dm)
    """
    return M*pin.exp(dm)


def update_est(X: List[pin.SE3], dx: np.ndarray) -> List[pin.SE3]:
    """
    Update estimated poses.
    Inputs:
        X: list of object pose object variables
        dx: update step as an array of se3 tangent space "deltas"
    Returns:
        list of updated object poses X*exp(dx)
    """
    assert 6*len(X) == len(dx)
    return [rplus_se3(M,dx[6*i:6*i+6]) for i, M in enumerate(X)]
