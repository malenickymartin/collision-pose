import numpy as np
import pinocchio as pin
from src.optim_tools import error_se3, error_r3_so3


def num_diff_perc(M: pin.SE3, Mm: pin.SE3, err=error_se3, exp=pin.exp6, EPS=1e-6):
    e0 = err(M, Mm)
    J = np.zeros((6,6))
    for i in range(6):
        e = np.zeros(6)
        e[i] = EPS
        J[:,i] = (err(M*exp(e),Mm) - e0)/EPS
    
    return J

for _ in range(1000):
    M, Mm = pin.SE3.Random(), pin.SE3.Random()

    # SE3
    J_n = num_diff_perc(M, Mm, err=error_se3)
    e, J = error_se3(M, Mm, jac=True)

    assert np.allclose(error_se3(M,M), np.zeros(6), atol=1e-5)
    assert np.allclose(J, J_n, atol=1e-5)

    # R3xSO3
    J_n = num_diff_perc(M, Mm, err=error_r3_so3)
    e, J = error_r3_so3(M, Mm, jac=True)

    assert np.allclose(error_r3_so3(M,M), np.zeros(6), atol=1e-5)
    assert np.allclose(J, J_n, atol=1e-5)



