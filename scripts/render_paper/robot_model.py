import pinocchio as pin
import numpy as np
import panda_py

T_RC = pin.SE3(np.array([[-0.010841866078458162, -0.9999289185510865, -0.004961026626792572, 0.04390298289963003],
                         [0.9999362837135557, -0.010826081565355139, -0.0031975726831303805, -0.03343932125362424],
                         [0.0031436369161215024, -0.004995378183405597, 0.999982581720174, 0.07119525019416911],
                         [0.0, 0.0, 0.0, 1.0]]))
T_BR_init = pin.SE3(np.array([[0.9994643006649909, 0.025898287627286375, -0.020009757475178653, 0.6024427213699449],
                              [0.026147555370427115, -0.999582448548234, 0.012297719401093822, 0.005938269974116062],
                              [-0.019682912497686254, -0.01281433776251958, -0.9997241498049931, 0.6873787949264959],
                              [0.0, 0.0, 0.0, 1.0]]))

class RobotModel():
    def __init__(self, data_dir, eef_id = 'panda_hand'):
        urdf_path = data_dir / "panda_description" / "urdf" / "panda.urdf"
        self.robot_model = pin.buildModelFromUrdf(str(urdf_path))
        self.robot_data = self.robot_model.createData()
        self.robod_eef_id = self.robot_model.getFrameId(eef_id)
        self.joint_offsets = np.array([ 0.0208242,  -0.00844751, -0.00179873,  0.00482039,  0.01542156, -0.01619169, -0.01002264])
 
    def get_cam_pose(self, q_7):
        corrected_joint_values = q_7 + self.joint_offsets
        expanded_joint_values = np.concatenate((corrected_joint_values, np.array([0, 0])))
        pin.forwardKinematics(self.robot_model, self.robot_data, expanded_joint_values)
        T_BR = pin.updateFramePlacement(self.robot_model, self.robot_data, self.robod_eef_id)
        T_BC = T_BR * T_RC
        return T_BC
    
    def get_robot_pose(self, q_7):
        corrected_joint_values = q_7 + self.joint_offsets
        expanded_joint_values = np.concatenate((corrected_joint_values, np.array([0, 0])))
        pin.forwardKinematics(self.robot_model, self.robot_data, expanded_joint_values)
        T_BR = pin.updateFramePlacement(self.robot_model, self.robot_data, self.robod_eef_id)
        return T_BR
        
    def ik_num(self, q_start_7: list, pose_goal: pin.SE3, eps=1e-3):
        model = self.robot_model
        data  = model.createData()

        q_start_9 = np.concatenate([q_start_7, [0, 0]])

        FRAME_ID = model.getFrameId("panda_hand_tcp")
        oMdes = pose_goal

        q      = np.array(q_start_9)
        IT_MAX = 2000
        DT     = 3e-1
        damp   = 1e-3
        alpha = 1e-6

        i=0
        while True:
            oMf = panda_py.fk(q[:7])
            oMf = pin.SE3(oMf[:3,:3], oMf[:3,3])
            iMd = oMf.actInv(oMdes)
            err_T = pin.log(iMd).vector  # in joint frame
            err_q = alpha*(q-q_start_9)
            err = np.concatenate([err_T, err_q])
            if np.linalg.norm(err_T) < eps:
                success = True
                break
            if i >= IT_MAX:
                success = False
                break
            J = pin.computeFrameJacobian(model, data, q, FRAME_ID)  # in joint frame
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            J = np.concatenate([J, alpha*np.eye(9)])
            v = - J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(15), err))
            q = pin.integrate(model,q,v*DT)
            i += 1

        if success:
            print("Convergence achieved!")
            return q
        else:
            print("\nWarning: the iterative algorithm has not reached convergence to the desired precision")
            return np.array([])