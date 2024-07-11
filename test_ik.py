import pybullet as p
import numpy as np
import time
from scipy.spatial.transform import Rotation as sciR
import sys

from my_utils.utils_calc import *
from fanuc_leaphand_pybullet import FanucLeapHandPybullet


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
            




# ------------------------------------------------
if __name__ == '__main__':
    urdf_path = "./urdf/fanuc_leaphand.urdf"

    try:
        if sys.argv[1] == "global_rotation":
            test1(urdf_path)  # global rotation
        elif sys.argv[1] == "v_shape_rotation":
            test2(urdf_path)  # V-shape rotation
        else:
            raise ValueError(f"Invalid argv[1] {sys.argv[1]}. Must be 'global_rotation' or 'v_shape_rotation'.")
    except:
        raise ValueError("argv[1] must be 'global_rotation' or 'v_shape_rotation'.")