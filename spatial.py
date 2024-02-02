from copy import deepcopy
import numpy as np
import pinocchio as pin


def normalize_se3(M):
    pose = pin.SE3ToXYZQUAT(M)
    q_norm = np.linalg.norm(pose[3:])
    pose[3:] = pose[3:] / q_norm
    return pin.XYZQUATToSE3(pose)


def perturb_se3(M_lst, sig_t=0.0, sig_o=0.0):
    M_lst_pert = deepcopy(M_lst)
    for i in range(len(M_lst)):
        M_lst_pert[i].translation += np.random.normal(0, sig_t*np.ones(3))
        M_lst_pert[i].rotation = M_lst_pert[i].rotation @ pin.exp3(np.random.normal(0, sig_o*np.ones(3)))
    return M_lst_pert

