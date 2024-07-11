import pybullet as p
import numpy as np
import os, time
import scipy
from scipy.spatial.transform import Rotation as sciR
from scipy.optimize import minimize, Bounds
from scipy.linalg import block_diag

from my_utils import utils_calc
from my_utils.utils_calc import *



# ------------------------------------------------
class FanucLeapHandPybullet():
    def __init__(self, urdf_path, use_gui=True):

        self._physics_client_id = p.connect(p.GUI) if use_gui else p.connect(p.DIRECT)

        # Set the GUI camera parameters
        camera_distance = 2.0
        camera_yaw = 45
        camera_pitch = -30
        camera_target_position = [0, 0, 0]
        # Adjust the GUI viewpoint
        p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)
        
        # load URDF
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE
                # p.URDF_USE_SELF_COLLISION
        self.robot_id = p.loadURDF(urdf_path,
                                   basePosition=[0.0, 0.0, 0.0], 
                                   useFixedBase=True, flags=flags,
                                   physicsClientId=self._physics_client_id)
        
        self.getBasicInfo()
        

    # ----------------------------------
    def getBasicInfo(self):
        self.part_joints_name = {}
        self.part_joints_name["arm"] = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
        self.part_joints_name["finger0"] = ["1", "0", "2", "3"]
        self.part_joints_name["finger1"] = ["5", "4", "6", "7"]
        self.part_joints_name["finger2"] = ["9", "8", "10", "11"]
        self.part_joints_name["thumb"] = ["12", "13", "14", "15"] 
        self.part_joints_name["hand"] = self.part_joints_name["finger0"] + \
                                        self.part_joints_name["finger1"] + \
                                        self.part_joints_name["finger2"] + \
                                        self.part_joints_name["thumb"]

        self.tcp_link_name = {}
        self.tcp_link_name["arm"] = "palm_lower"
        self.tcp_link_name["finger0"] = "finger0_tip_center"
        self.tcp_link_name["finger1"] = "finger1_tip_center"
        self.tcp_link_name["finger2"] = "finger2_tip_center"
        self.tcp_link_name["thumb"] = "thumb_tip_center"

        self.num_joints = p.getNumJoints(self.robot_id, physicsClientId=self._physics_client_id)
        self.joint_name_to_ids = {}
        self.joint_name_to_limits = {}
        self.link_name_to_ids = {}
        self.joint_id_to_names = [""] * self.num_joints
        self.joint_id_to_lower_limits = [0] * self.num_joints
        self.joint_id_to_upper_limits = [0] * self.num_joints
        self.active_joint_ids = []

        for id in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, id, physicsClientId=self._physics_client_id)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]
            joint_lower_limit = joint_info[8]
            joint_upper_limit = joint_info[9]
            link_name = joint_info[12].decode("UTF-8")

            # mannually modify some of the joint bounds
            if joint_name == "0":
                joint_lower_limit = -np.pi / 4
                joint_upper_limit = np.pi / 4

            self.joint_name_to_ids[joint_name] = id
            self.joint_name_to_limits[joint_name] = (joint_lower_limit, joint_upper_limit)
            self.joint_id_to_names[id] = joint_name
            self.joint_id_to_lower_limits[id] = joint_lower_limit
            self.joint_id_to_upper_limits[id] = joint_upper_limit
            self.link_name_to_ids[link_name] = id

            if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
                self.active_joint_ids.append(id)

        self.part_joints_id = {}
        for part_name in list(self.part_joints_name.keys()):
            joints_name = self.part_joints_name[part_name]
            joints_id = [self.joint_name_to_ids[name]  for name in joints_name]
            self.part_joints_id[part_name] = joints_id

        self.tcp_link_ids = {}
        for finger_name in list(self.tcp_link_name.keys()):
            link_name = self.tcp_link_name[finger_name]
            self.tcp_link_ids[finger_name] = self.link_name_to_ids[link_name]

        # print("num of all joints: ", self.num_joints)


    # ----------------------------------
    """
        return: 
            joint pos of all joints, including active joints and fixed joints
    """
    def getAllJointPos(self):
        all_states = p.getJointStates(self.robot_id, range(self.num_joints), physicsClientId=self._physics_client_id)
        all_joints_pos = [one_joint_state[0] for one_joint_state in all_states]
        return np.array(all_joints_pos)
    

    # ----------------------------------
    def checkJointDim(self, part_name, joints):
        joints = np.array(joints)
        if joints.shape[0] != len(self.part_joints_id[part_name]):
            raise NameError("Wrong dimension of input joints !")
    

    # ----------------------------------
    def setJointPos(self, part_name, joint_pos):
        self.checkJointDim(part_name, joint_pos)
        for pos, joint_id in zip(joint_pos, self.part_joints_id[part_name]):
            p.resetJointState(self.robot_id, joint_id, pos, physicsClientId=self._physics_client_id)
    

    # ----------------------------------
    def getTcpGlobalPose(self, part_name, arm_joint_pos=None, finger_joint_pos=None, local_position=None):
        if arm_joint_pos is not None:
            self.setJointPos("arm", arm_joint_pos)
        if part_name != "arm" and finger_joint_pos is not None:
            self.setJointPos(part_name, finger_joint_pos)

        b_fk = (arm_joint_pos is not None) or (finger_joint_pos is not None)

        # get the position of the tcp link
        tcp_link_id = self.tcp_link_ids[part_name]
        state = p.getLinkState(self.robot_id, tcp_link_id, computeLinkVelocity=0,
                               computeForwardKinematics=b_fk, physicsClientId=self._physics_client_id)
        pos, quat = state[4], state[5] # worldLinkFramePosition, worldLinkFrameOrientation

        # get the position of the target point
        if local_position is not None:
            pos = sciR.from_quat(quat).as_matrix() @ np.array(local_position).reshape(-1, 1) \
                + np.array(pos).reshape(-1, 1)
        
        return np.array(pos).reshape(-1, ), np.array(quat)
    

    # ----------------------------------
    def getFingerTcpLocalPose(self, part_name, finger_joint_pos=None, local_position=None):
        if part_name == "arm":
            raise NameError("The input part_name should not be 'arm' !")
        
        # get finger tcp global pose
        finger_pos, finger_quat = self.getTcpGlobalPose(part_name, arm_joint_pos=None, finger_joint_pos=finger_joint_pos,
                                                        local_position=local_position)
        # get arm tcp global pose
        arm_tcp_pos, arm_tcp_quat = self.getTcpGlobalPose("arm", arm_joint_pos=None, finger_joint_pos=None)

        # calculate the finger tcp local pose
        finger_tcp_pose_in_base = utils_calc.posQuat2Isometry3d(finger_pos, finger_quat)
        arm_tcp_pose_in_base = utils_calc.posQuat2Isometry3d(arm_tcp_pos, arm_tcp_quat)
        finger_tcp_pose_in_arm_tcp = np.linalg.inv(arm_tcp_pose_in_base) @ finger_tcp_pose_in_base
        pos, quat = utils_calc.isometry3dToPosQuat(finger_tcp_pose_in_arm_tcp)
        
        return np.array(pos), np.array(quat)
    

    # ----------------------------------
    def getArmTwoFingersPose(self, 
                             finger0_name, finger1_name, 
                             arm_joint_pos, finger0_joint_pos, finger1_joint_pos,
                             arm_local_position, finger0_local_position, finger1_local_position):
        # get arm tcp global pose
        arm_tcp_pos, arm_tcp_quat = self.getTcpGlobalPose("arm", 
                                                          arm_joint_pos=arm_joint_pos,
                                                          local_position=None)
        # get finger tcp global pose
        finger0_pos, finger0_quat = self.getTcpGlobalPose(finger0_name,
                                                          local_position=finger0_local_position)
        finger1_pos, finger1_quat = self.getTcpGlobalPose(finger1_name,
                                                          local_position=finger1_local_position)
        
        arm_tcp_pose_in_base = utils_calc.posQuat2Isometry3d(arm_tcp_pos, arm_tcp_quat)
        finger0_tcp_pose_in_base = utils_calc.posQuat2Isometry3d(finger0_pos, finger0_quat)
        finger1_tcp_pose_in_base = utils_calc.posQuat2Isometry3d(finger1_pos, finger1_quat)

        finger0_tcp_pose_in_arm_tcp = np.linalg.inv(arm_tcp_pose_in_base) @ finger0_tcp_pose_in_base
        finger1_tcp_pose_in_arm_tcp = np.linalg.inv(arm_tcp_pose_in_base) @ finger1_tcp_pose_in_base

        return arm_tcp_pose_in_base, finger0_tcp_pose_in_base, finger1_tcp_pose_in_base, \
                finger0_tcp_pose_in_arm_tcp, finger1_tcp_pose_in_arm_tcp


    # ----------------------------------
    def getGlobalJacobian(self, part_name, arm_joint_pos=None, finger_joint_pos=None, local_position=None):
        """
            Input:
                local_position: target point position in the fingertip center frame
        """
        all_joint_pos = self.getAllJointPos()

        if arm_joint_pos is not None:
            self.checkJointDim("arm", arm_joint_pos)
            all_joint_pos[self.part_joints_id["arm"]] = np.array(arm_joint_pos)
        if part_name != "arm" and finger_joint_pos is not None:
            self.checkJointDim(part_name, finger_joint_pos)
            all_joint_pos[self.part_joints_id[part_name]] = np.array(finger_joint_pos)

        if local_position is None:
            local_position = np.array([0, 0, 0])
        else:
            local_position = np.array(local_position)

        active_joint_pos = all_joint_pos[self.active_joint_ids]
        num_active_joints = active_joint_pos.shape[0]

        tcp_link_id = self.tcp_link_ids[part_name]
        linear_jaco, angu_jaco = p.calculateJacobian(self.robot_id, 
                                                     tcp_link_id, 
                                                     localPosition=local_position.tolist(), 
                                                     objPositions=active_joint_pos.tolist(), 
                                                     objVelocities=[0]*num_active_joints,
                                                     objAccelerations=[0]*num_active_joints)
        active_jacobian_matrix = np.vstack([linear_jaco, angu_jaco]) # 6 * n_active_joints

        all_jacobian_matrix = np.zeros((6, self.num_joints)) # 6 * n_all_joints
        all_jacobian_matrix[:, self.active_joint_ids] = active_jacobian_matrix

        if part_name == "arm":
            joints_id = self.part_joints_id["arm"]
        else:
            joints_id = self.part_joints_id["arm"] + self.part_joints_id[part_name]
        part_jacobian_matrix = all_jacobian_matrix[:, joints_id] # 6 * n_arm_joints

        return part_jacobian_matrix
    

    # ----------------------------------
    def getFingerLocalJacobian(self, part_name, finger_joint_pos=None, local_position=None):
        """
            return:
                the returned Jacobian is defined in the arm tcp frame
        """
        if part_name == "arm":
            raise NameError("The input part_name should not be 'arm' !")
        
        global_jacobian = self.getGlobalJacobian(part_name, finger_joint_pos=finger_joint_pos,
                                                 local_position=local_position)

        # remove the columns related to the arm joints
        finger_jacobian_in_base = global_jacobian[:, len(self.part_joints_id["arm"]) : ]  

        # transform the Jacobian from base frame to arm tcp frame
        _, arm_tcp_quat_in_base = self.getTcpGlobalPose("arm")
        base_rotmat_in_arm_tcp = sciR.from_quat(arm_tcp_quat_in_base).as_matrix().T
        local_jacobian = block_diag(base_rotmat_in_arm_tcp, base_rotmat_in_arm_tcp) @ finger_jacobian_in_base

        return local_jacobian
    

    # ----------------------------------
    def fingerGlobalJacoToLocalJaco(self, global_jaco, arm_pose):
        # remove the columns related to the arm joints
        finger_jacobian_in_base = global_jaco[:, len(self.part_joints_id["arm"]) : ]  

        # transform the Jacobian from base frame to arm tcp frame
        base_rotmat_in_arm_tcp = arm_pose[0:3, 0:3].T
        local_jaco = block_diag(base_rotmat_in_arm_tcp, base_rotmat_in_arm_tcp) @ finger_jacobian_in_base
        return local_jaco
    

    # ----------------------------------
    def globalArmTwoFingerIKSQP(self,
                            finger0_name, finger1_name,
                            arm_target_pose=None,
                            finger0_target_pose=None, 
                            finger1_target_pose=None,
                            finger0_target_local_pose=None,
                            finger1_target_local_pose=None,
                            fingers_target_relative_pos=None,
                            weights=None,
                            arm_joint_pos_init=None, finger0_joint_pos_init=None, finger1_joint_pos_init=None,
                            finger0_local_position=None, finger1_local_position=None):
        
        if arm_target_pose is not None:
            arm_target_pos, arm_target_ori = isometry3dToPosOri(arm_target_pose)
        if finger0_target_pose is not None:
            finger0_target_pos, finger0_target_ori = isometry3dToPosOri(finger0_target_pose)
        if finger1_target_pose is not None:
            finger1_target_pos, finger1_target_ori = isometry3dToPosOri(finger1_target_pose)
        if finger0_target_local_pose is not None:
            finger0_target_local_pos, finger0_target_local_ori = isometry3dToPosOri(finger0_target_local_pose)
        if finger1_target_local_pose is not None:
            finger1_target_local_pos, finger1_target_local_ori = isometry3dToPosOri(finger1_target_local_pose)
        
        self.curr_arm_pose = None
        self.curr_err = None

        # ------------------------
        def calcTargetError(arm_joint_pos, finger0_joint_pos, finger1_joint_pos):
            t1 = time.time()

            self.setJointPos("arm", arm_joint_pos)
            self.setJointPos(finger0_name, finger0_joint_pos)
            self.setJointPos(finger1_name, finger1_joint_pos)

            arm_pose, finger0_pose, finger1_pose, finger0_local_pose, finger1_local_pose \
                = self.getArmTwoFingersPose(finger0_name, finger1_name,
                                              arm_joint_pos, finger0_joint_pos, finger1_joint_pos,
                                              None, finger0_local_position, finger1_local_position)
            err_list = []
            if arm_target_pose is not None:
                arm_pos_err = arm_target_pos - arm_pose[0:3, 3].reshape(-1,)
                arm_ori_err = (arm_target_ori * sciR.from_matrix(arm_pose[0:3, 0:3].T)).as_rotvec()
                err_list.extend([arm_pos_err, arm_ori_err])
            if finger0_target_pose is not None:
                finger0_pos_err = finger0_target_pos - finger0_pose[0:3, 3].reshape(-1,)
                finger0_ori_err = (finger0_target_ori * sciR.from_matrix(finger0_pose[0:3, 0:3].T)).as_rotvec()
                err_list.extend([finger0_pos_err, finger0_ori_err])
            if finger1_target_pose is not None:
                finger1_pos_err = finger1_target_pos - finger1_pose[0:3, 3].reshape(-1,)
                finger1_ori_err = (finger1_target_ori * sciR.from_matrix(finger1_pose[0:3, 0:3].T)).as_rotvec()
                err_list.extend([finger1_pos_err, finger1_ori_err])
            if finger0_target_local_pose is not None:
                finger0_local_pos_err = finger0_target_local_pos - finger0_local_pose[0:3, 3].reshape(-1,)
                finger0_local_ori_err = (finger0_target_local_ori * sciR.from_matrix(finger0_local_pose[0:3, 0:3].T)).as_rotvec()
                err_list.extend([finger0_local_pos_err, finger0_local_ori_err])
            if finger1_target_local_pose is not None:
                finger1_local_pos_err = finger1_target_local_pos - finger1_local_pose[0:3, 3].reshape(-1,)
                finger1_local_ori_err = (finger1_target_local_ori * sciR.from_matrix(finger1_local_pose[0:3, 0:3].T)).as_rotvec()
                err_list.extend([finger1_local_pos_err, finger1_local_ori_err])
            if fingers_target_relative_pos is not None:
                fingers_relative_pos_err = fingers_target_relative_pos - (finger0_local_pose[0:3, 3] - finger1_local_pose[0:3, 3]).reshape(-1,)
                err_list.append(fingers_relative_pos_err)

            err = np.concatenate(err_list, axis=0).reshape(-1, 1)
            self.curr_err = err.copy()
            self.curr_arm_pose = arm_pose.copy()

            return err

        # ------------------------
        def objectFunction(joint_pos):
            arm_joint_pos, finger0_joint_pos, finger1_joint_pos = joint_pos[0:6], joint_pos[6:10], joint_pos[10:14]
            err = calcTargetError(arm_joint_pos, finger0_joint_pos, finger1_joint_pos)
            cost = 1.0/2.0 * err.T @ weights @ err
            return cost[0, 0]
        
        # ------------------------
        def objectJacobian(joint_pos):
            arm_joint_pos, finger0_joint_pos, finger1_joint_pos = joint_pos[0:6], joint_pos[6:10], joint_pos[10:14]

            t1 = time.time()

            pybullet_joint_pos = self.getAllJointPos()[self.part_joints_id["arm"] + self.part_joints_id[finger0_name] + self.part_joints_id[finger1_name]]
            if np.linalg.norm(joint_pos - pybullet_joint_pos) > 1e-8:
                err = calcTargetError(arm_joint_pos, finger0_joint_pos, finger1_joint_pos)
            else:
                err = self.curr_err.copy()

            # get all jacobians
            finger0_global_jaco = self.getGlobalJacobian(finger0_name, local_position=finger0_local_position)
            finger1_global_jaco = self.getGlobalJacobian(finger1_name, local_position=finger1_local_position)
            finger0_local_jaco = self.fingerGlobalJacoToLocalJaco(finger0_global_jaco, self.curr_arm_pose.copy())
            finger1_local_jaco = self.fingerGlobalJacoToLocalJaco(finger1_global_jaco, self.curr_arm_pose.copy())

            jaco_list = []
            if arm_target_pose is not None:
                arm_jaco = self.getGlobalJacobian(part_name="arm")
                jaco_list.append( np.hstack([arm_jaco, np.zeros((6, 4)), np.zeros((6, 4))]) )
            if finger0_target_pose is not None:
                jaco_list.append( np.hstack([finger0_global_jaco,  np.zeros((6, 4))]) )
            if finger1_target_pose is not None:
                jaco_list.append( np.hstack([finger1_global_jaco[:, 0:6], np.zeros((6, 4)), finger1_global_jaco[:, 6:10]]) )
            if finger0_target_local_pose is not None:
                jaco_list.append( np.hstack([np.zeros((6, 6)), finger0_local_jaco, np.zeros((6, 4))]) )
            if finger1_target_local_pose is not None:
                jaco_list.append( np.hstack([np.zeros((6, 6)), np.zeros((6, 4)), finger1_local_jaco]) )
            if fingers_target_relative_pos is not None:
                jaco_list.append( np.hstack([np.zeros((3, 6)), finger0_local_jaco[0:3, :], -finger1_local_jaco[0:3, :]]) )
            jaco = np.vstack(jaco_list)
            
            object_jaco = err.T @ weights @ (-jaco)
            return object_jaco.reshape(-1, )

        # initial value
        joint_pos_init = np.concatenate([arm_joint_pos_init, finger0_joint_pos_init, finger1_joint_pos_init], axis=0)
        # bounds
        joint_pos_lb = np.array(self.joint_id_to_lower_limits)[self.part_joints_id["arm"] + self.part_joints_id["finger0"] + self.part_joints_id["thumb"]]
        joint_pos_ub = np.array(self.joint_id_to_upper_limits)[self.part_joints_id["arm"] + self.part_joints_id["finger0"] + self.part_joints_id["thumb"]]
        joint_pos_bounds = [(joint_pos_lb[i], joint_pos_ub[i]) for i in range(joint_pos_lb.shape[0])]

        res = minimize(fun=objectFunction, 
                       x0=joint_pos_init, 
                       jac=objectJacobian, 
                       bounds=joint_pos_bounds,
                       method='SLSQP', 
                        options={'ftol':1e-8, 'disp': False, 'maxiter': 1000}
                       )  
        res_joint_pos = res.x.reshape(-1, )

        res_arm_joint_pos = res_joint_pos[0:6]
        res_finger0_joint_pos = res_joint_pos[6:10]
        res_finger1_joint_pos = res_joint_pos[10:14]

        err = calcTargetError(res_arm_joint_pos, res_finger0_joint_pos, res_finger1_joint_pos)
        weighted_err = np.sqrt(err.T @ weights @ err)

        return res_arm_joint_pos, res_finger0_joint_pos, res_finger1_joint_pos, weighted_err


# --------------------------------------------------------------------------------------------
def test1(urdf_path):
    """
        test of 'Global Rotation' in Fig.6 of the paper
    """
    robot = FanucLeapHandPybullet(urdf_path=urdf_path, use_gui=True)

    arm_joint_pos = [0, 0, 0, 0, -np.pi/2, 0]
    finger0_joint_pos = [1.8714565,   0.11044662, -0.48933986,  0.15493207]
    thumb_joint_pos = [1.57079637,  0.00460194,  0.61359233, -0.6304661]

    robot.setJointPos("arm", arm_joint_pos)
    robot.setJointPos("finger0", finger0_joint_pos)
    robot.setJointPos("thumb", thumb_joint_pos)

    finger0_pos, finger0_quat = robot.getTcpGlobalPose("finger0")

    finger0_local_pos, finger0_local_quat = robot.getFingerTcpLocalPose("finger0")
    thumb_local_pos, thumb_local_quat = robot.getFingerTcpLocalPose("thumb")

    for i in range(5):  # 15 / 30/ 45 / 60 / 75 degree
        finger0_quat = (sciR.from_quat(finger0_quat) * sciR.from_euler("ZXY", [-np.pi/12, 0, 0])).as_quat()

        weights = np.diag([0, 0, 0, 5e-2, 5e-2, 0,
                        100, 100, 100, 1, 1, 1,
                            0, 0, 0, 0, 1e-1, 1e-1,
                            0, 0, 0, 0, 1e-1, 1e-1,
                            10, 10, 10])
        
        t1 = time.time()
        res_arm_joint_pos, res_finger0_joint_pos, res_thumb_joint_pos, weighted_err = robot.globalArmTwoFingerIKSQP(
                                                finger0_name="finger0", finger1_name="thumb",
                                                arm_target_pose=posQuat2Isometry3d([0,0,0], [0,0,0,1]),
                                                finger0_target_pose=posQuat2Isometry3d(finger0_pos, finger0_quat),
                                                finger1_target_pose=None,
                                                finger0_target_local_pose=posQuat2Isometry3d(finger0_local_pos, finger0_local_quat),
                                                finger1_target_local_pose=posQuat2Isometry3d(thumb_local_pos, thumb_local_quat),
                                                fingers_target_relative_pos=(finger0_local_pos - thumb_local_pos),
                                                weights=weights,
                                                arm_joint_pos_init=arm_joint_pos,
                                                finger0_joint_pos_init=finger0_joint_pos,
                                                finger1_joint_pos_init=thumb_joint_pos,
                                                finger0_local_position=[0, 0, 0], finger1_local_position=[0,0,0]
                                                )
        
        print(f"Global rotation angle {15*(i+1)}, time cost {time.time() - t1} s, weighted error {weighted_err.item()}")

        arm_joint_pos = res_arm_joint_pos
        finger0_joint_pos = res_finger0_joint_pos
        thumb_joint_pos = res_thumb_joint_pos

        robot.setJointPos("arm", res_arm_joint_pos)
        robot.setJointPos("finger0", res_finger0_joint_pos)
        robot.setJointPos("thumb", res_thumb_joint_pos)

        time.sleep(2.0)
    
        
# --------------------------------------------------------------------------------------------
def test2(urdf_path):
    """
        test of 'V-Shape Rotation' in Fig.6 of the paper
    """
    robot = FanucLeapHandPybullet(urdf_path=urdf_path, use_gui=True)

    arm_joint_pos = [0, 0, 0, 0, -np.pi/2, 0]
    finger0_joint_pos = [1.8714565,   0.11044662, -0.48933986,  0.15493207]
    thumb_joint_pos = [1.57079637,  0.00460194,  0.61359233, -0.6304661]

    robot.setJointPos("arm", arm_joint_pos)
    robot.setJointPos("finger0", finger0_joint_pos)
    robot.setJointPos("thumb", thumb_joint_pos)

    norminal_finger0_local_pos, norminal_finger0_local_quat = robot.getFingerTcpLocalPose("finger0")
    norminal_thumb_local_pos, norminal_thumb_local_quat = robot.getFingerTcpLocalPose("thumb")
    norminal_finger0_global_pos, norminal_finger0_global_quat = robot.getTcpGlobalPose("finger0")
    arm_pos, arm_quat = robot.getTcpGlobalPose("arm")

    local_position = [0, -0.012, 0]
    target_finger_angle = -np.deg2rad(20)
    target_grip_angle = 0

    for i in range(5):  # 10 / 20/ 30 / 40 / 50 degree
        target_grip_angle += np.deg2rad(10)

        target_finger0_local_pos = norminal_finger0_local_pos
        target_thumb_local_pos = norminal_thumb_local_pos

        target_finger0_local_quat = (sciR.from_euler('xyz', [target_finger_angle, 0, 0]) \
                                        * sciR.from_quat(norminal_finger0_local_quat) \
                                        * sciR.from_euler('XYZ', 
                                            [-target_grip_angle, 0, 0])).as_quat()
        target_thumb_local_quat = (sciR.from_euler('xyz', [target_finger_angle, 0, 0]) \
                                        * sciR.from_quat(norminal_thumb_local_quat) \
                                        * sciR.from_euler('XYZ', 
                                            [-target_grip_angle, 0, 0])).as_quat()
        
        target_fingers_relative_pos = [0, 0, 0]

        weights = np.diag([0, 0, 0, 10, 10, 10,
                          100, 100, 100, 0, 0, 0,
                            1, 0, 0, 1e-1, 1e-1, 1e-2,
                            1, 0, 0, 1e-1, 1e-1, 1e-2,
                            100, 100, 100])

        t1 = time.time()

        res_arm_joint_pos, res_finger0_joint_pos, res_thumb_joint_pos, weighted_err = robot.globalArmTwoFingerIKSQP(
                    finger0_name="finger0", finger1_name="thumb",
                    arm_target_pose=posQuat2Isometry3d(arm_pos, arm_quat),
                    finger0_target_pose=posQuat2Isometry3d(norminal_finger0_global_pos, norminal_finger0_global_quat),
                    finger1_target_pose=None,
                    finger0_target_local_pose=posQuat2Isometry3d(target_finger0_local_pos, target_finger0_local_quat),
                    finger1_target_local_pose=posQuat2Isometry3d(target_thumb_local_pos, target_thumb_local_quat),
                    fingers_target_relative_pos=target_fingers_relative_pos,
                    weights=weights,
                    arm_joint_pos_init=arm_joint_pos,
                    finger0_joint_pos_init=finger0_joint_pos,
                    finger1_joint_pos_init=thumb_joint_pos,
                    finger0_local_position=local_position, finger1_local_position=local_position
                    )
            
        print(f"V-shape rotation angle {15*(i+1)}, time cost {time.time() - t1} s, weighted error {weighted_err.item()}")
        
        arm_joint_pos = res_arm_joint_pos
        finger0_joint_pos = res_finger0_joint_pos
        thumb_joint_pos = res_thumb_joint_pos

        robot.setJointPos("arm", res_arm_joint_pos)
        robot.setJointPos("finger0", res_finger0_joint_pos)
        robot.setJointPos("thumb", res_thumb_joint_pos)

        time.sleep(2.0)
            