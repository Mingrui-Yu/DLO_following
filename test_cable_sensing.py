import time
import sys
import numpy as np
from my_utils.utils_calc import *
import imageio.v2 as imageio
from fanuc_leaphand_pybullet import FanucLeapHandPybullet
from cable_sensing import CableSensing


class LeapHandNorminal(object):
    """
        We mannually define a reference configuration of the leap hand.
        Since the FK of the hand is not very accurate, we manually define a reference hand configuration where the index and thumb finger tip contacts each other in parallel.
        We rectify the relative poses of the two fingers calculated by FK using this reference configuration.
        Below 'norminal' refers to the 'reference' in the paper.
    """
    def __init__(self, pybullet):
        
        self.norminal_joint_pos = np.array([1.79168952,  0.03067962, -0.10431069, -0.11811652,  
                                            0, 0, 0, 0,  
                                            0, 0, 0, 0,
                                            1.57079637, -0.01073787, 0.44485444, -0.39269909])

        self.finger0_norminal_joint_pos = self.norminal_joint_pos[0:4]
        self.thumb_norminal_joint_pos = self.norminal_joint_pos[12:16]

        P = pybullet
        local_position = [0, 0, 0]
        self.norminal_finger0_tcp_pos, self.norminal_finger0_tcp_quat = P.getFingerTcpLocalPose("finger0", 
                                                                finger_joint_pos=self.finger0_norminal_joint_pos, 
                                                                local_position=local_position)
        self.norminal_thumb_tcp_pos, self.norminal_thumb_tcp_quat = P.getFingerTcpLocalPose("thumb", 
                                                                finger_joint_pos=self.thumb_norminal_joint_pos, 
                                                                local_position=local_position)
        
        norminal_finger0_pose = posQuat2Isometry3d(self.norminal_finger0_tcp_pos, self.norminal_finger0_tcp_quat)
        norminal_thumb_pose = posQuat2Isometry3d(self.norminal_thumb_tcp_pos, self.norminal_thumb_tcp_quat)

        real_norminal_thumb_pose_in_finger0 = np.linalg.inv(norminal_finger0_pose) @ norminal_thumb_pose
        pos, quat = isometry3dToPosQuat(real_norminal_thumb_pose_in_finger0)
        expect_norminal_thumb_pose_in_finger0 = posQuat2Isometry3d([0, 0, 0], quat)  # we regard the position of the index and thumb fingertip to be the same at the reference configuration

        self.compensate_transform = norminal_finger0_pose @ expect_norminal_thumb_pose_in_finger0 @ np.linalg.inv(norminal_thumb_pose)
    

    def rectifyThumbPoseInFinger0(self, finger0_pose, thumb_pose):
        """
            We rectify the relative poses of the two fingers calculated by FK using this reference configuration.
        """
        thumb_pose_in_finger0 = np.linalg.inv(finger0_pose) @ self.compensate_transform @ thumb_pose
        return thumb_pose_in_finger0
    


# ------------------------------------
if __name__ == '__main__':

    if len(sys.argv) < 2:
        raise ValueError("argv[1] must be 'seq_0' or 'seq_1'.")

    if sys.argv[1] == "seq_0":
        data_dir = "./data/cable_sensing/0/"
        open3d_viewpoint = 0
    elif sys.argv[1] == "seq_1":
        data_dir = "./data/cable_sensing/1/"
        open3d_viewpoint = 1
    else:
        raise ValueError("argv[1] must be 'seq_0' or 'seq_1'.")


    urdf_path = "./urdf/fanuc_leaphand.urdf"

    cable_sensing = CableSensing(b_visualize=True, open3d_viewpoint=open3d_viewpoint)

    P = FanucLeapHandPybullet(
            urdf_path=urdf_path,
            use_gui=False)
    
    leaphand_norminal = LeapHandNorminal(pybullet=P)

    # load data
    tactile_0_rgb_dir = data_dir + "tactile_0_rgb_img/"
    tactile_0_depth_dir = data_dir + "tactile_0_depth_img/"
    tactile_0_marker_dir = data_dir + "tactile_0_marker_pos/"
    tactile_0_marker_init_dir = data_dir + "tactile_0_marker_pos_init/"

    tactile_1_rgb_dir = data_dir + "tactile_1_rgb_img/"
    tactile_1_depth_dir = data_dir + "tactile_1_depth_img/"
    tactile_1_marker_dir = data_dir + "tactile_1_marker_pos/"
    tactile_1_marker_init_dir = data_dir + "tactile_1_marker_pos_init/"

    data_dict = np.load(data_dir + "data.npy", allow_pickle=True).item()

    seq_hand_curr_joint_pos = data_dict["hand_curr_joint_pos"]
    seq_hand_target_joint_pos = data_dict["hand_target_joint_pos"]
    seq_timestamp = data_dict["tactile_timestamp"]

    idx = 0
    while True:
        try:
            tactile_0_depth_img = (imageio.imread(tactile_0_depth_dir + str(idx) + ".png") - 10000.0) / 1000.0 # cv2 cannot read uint16 image, using imageio instead
            tactile_1_depth_img = (imageio.imread(tactile_1_depth_dir + str(idx) + ".png") - 10000.0) / 1000.0
        except:
            idx = 0
            time.sleep(2.0)
            continue

        # finger0_curr_joint_pos = seq_hand_curr_joint_pos[idx][0:4]
        finger0_target_joint_pos = seq_hand_target_joint_pos[idx][0:4]
        # thumb_curr_joint_pos = seq_hand_curr_joint_pos[idx][12:16]
        thumb_target_joint_pos = seq_hand_target_joint_pos[idx][12:16]

        P.setJointPos("finger0", finger0_target_joint_pos)
        P.setJointPos("thumb", thumb_target_joint_pos)

        finger0_tcp_pos, finger0_tcp_quat = P.getFingerTcpLocalPose("finger0", finger_joint_pos=finger0_target_joint_pos)
        thumb_tcp_pos, thumb_tcp_quat = P.getFingerTcpLocalPose("thumb", finger_joint_pos=thumb_target_joint_pos)

        tactile_frame_0_pose = posQuat2Isometry3d(finger0_tcp_pos, finger0_tcp_quat)
        tactile_frame_1_pose = posQuat2Isometry3d(thumb_tcp_pos, thumb_tcp_quat) 
        rec_tactile_1_pose_in_tactile_0 = leaphand_norminal.rectifyThumbPoseInFinger0(tactile_frame_0_pose, tactile_frame_1_pose)

        t1 = time.time()
        inhand_dlo_pose = cable_sensing.getCableInhandState(depth_map_0=tactile_0_depth_img,
                                                   depth_map_1=tactile_1_depth_img,
                                                   tactile_1_pose_in_tactile_0=rec_tactile_1_pose_in_tactile_0
                                                   )
        
        print(f"Frame {idx},  cable sening time cost: {time.time() - t1} s")
        time.sleep(0.1)
       
        idx += 1