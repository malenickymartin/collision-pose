from typing import List
import numpy as np
import pinocchio as pin


def res_grad_se3(M: pin.SE3, Mm: pin.SE3):
    """
    Happypose measurement distance residual and gradient.

    M: estimated pose
    Mm: measured pose

    Return:
    - residual(M,Mm) = T - Tm := Log6(Tm.inv()@T)
    and 
    - gradient = res * dres/dM
    """
    Mrel = Mm.inverse()*M
    res = pin.log(Mrel).vector
    J = pin.Jlog6(Mrel)
    return res, res @ J


def perception_res_grad(M_lst: List[pin.SE3], Mm_lst: List[pin.SE3]):
    """
    Compute residuals and gradients for all pose estimates|measurement pairs.
    """
    assert len(M_lst) == len(Mm_lst)
    N = len(M_lst)
    grad = np.zeros(6*N)
    res = np.zeros(6*N)
    for i in range(N):
        res[6*i:6*i+6], grad[6*i:6*i+6] = res_grad_se3(M_lst[i], Mm_lst[i])
    
    return res, grad


def rplus_se3(M, dm):
    return M*pin.exp(dm)


def update_est(X: List[pin.SE3], dx: np.ndarray):
    """
    Update estimated poses.

    X: list of object pose object variables
    dx: update step as an array of se3 tangent space "deltas"
    """
    assert 6*len(X) == len(dx)
    return [rplus_se3(M,dx[6*i:6*i+6]) for i, M in enumerate(X)]
