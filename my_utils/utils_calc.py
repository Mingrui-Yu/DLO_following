import numpy as np
from scipy.spatial.transform import Rotation as sciR
import time


# ---------------------------------------
def getUniformRandomDouble(lb, ub):
    return lb + (ub - lb) * np.random.rand()


# ---------------------------------------
def getGaussianRandomDouble(mean, sigma):
    return mean + sigma * np.random.randn()


# ---------------------------------------
def twoVecAngle(vec0, vec1):
    return np.arctan2(np.linalg.norm(np.cross(vec0, vec1)), np.dot(vec0, vec1))


# ---------------------------------------
def quatWXYZ2XYZW(quat_wxyz):
    quat_wxyz = np.array(quat_wxyz)
    original_shape = quat_wxyz.shape
    quat_wxyz = quat_wxyz.reshape(-1, 4)

    quat_xyzw = quat_wxyz[:, [1, 2, 3, 0]]

    return quat_xyzw.reshape(original_shape)


# ---------------------------------------
def quatXYZW2WXYZ(quat_xyzw):
    quat_xyzw = np.array(quat_xyzw)
    original_shape = quat_xyzw.shape
    quat_xyzw = quat_xyzw.reshape(-1, 4)

    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]

    return quat_wxyz.reshape(original_shape)


def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a 3x3 rotation matrix.
    
    Parameters:
        q (array_like): Quaternion in the form [w, x, y, z].

    Returns:
        R (ndarray): 3x3 rotation matrix.
    """
    # Normalize quaternion
    q = q / np.linalg.norm(q)
    
    # Extract quaternion components
    x, y, z, w = q
    
    # Compute rotation matrix
    R = np.array([[1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                  [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                  [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]])
    
    return R


# ---------------------------------------
def posQuat2Isometry3d(pos, quat):
    # quat: [x, y, z, w]
    pos = np.array(pos)
    rot_mat = sciR.from_quat(quat).as_matrix()
    isometry3d = np.block([[rot_mat, pos.reshape(3, 1)], 
                           [np.zeros((1, 3)), 1]])
    
    return isometry3d


# ---------------------------------------
def isometry3dToPosQuat(T):
    if T.shape[0] != 4 or T.shape[1] != 4:
        raise NameError("isometry3dToPosQuat(): invalid input.")
    pos = T[0:3, 3].reshape(-1, )
    R = T[0:3, 0:3]
    quat = sciR.from_matrix(R).as_quat() # quat: [x, y, z, w]
    return pos, quat


# ---------------------------------------
def isometry3dToPosOri(T):
    if T.shape[0] != 4 or T.shape[1] != 4:
        raise NameError("isometry3dToPosQuat(): invalid input.")
    pos = T[0:3, 3].reshape(-1, )
    R = T[0:3, 0:3]
    return pos, sciR.from_matrix(R)


# ---------------------------------------
'''
    input:
        positions: size of [-1, 3]
        target_frame_pose: 
            matrix with size of [4, 4]
            target_frame_pose in current frame
    output:
        transformed_pos: size of [-1, 3]
'''
def transformPositions(positions, 
                       target_frame_pose=None, 
                       target_frame_pose_inv=None):
    if (target_frame_pose is None) and (target_frame_pose_inv is None):
        raise NameError("Both target_frame_pose and target_frame_pose_inv are None !")
    elif (target_frame_pose is not None) and (target_frame_pose_inv is not None):
        raise NameError("Both target_frame_pose and target_frame_pose_inv are not None !")

    positions = np.array(positions)
    original_shape = positions.shape
    argument_pos = positions.reshape(-1, 3)
    argument_pos = np.hstack([argument_pos, np.ones((argument_pos.shape[0], 1))])

    if target_frame_pose is not None:
        res = np.dot(np.linalg.inv(target_frame_pose), argument_pos.T)
    elif target_frame_pose_inv is not None:
        res = np.dot(target_frame_pose_inv, argument_pos.T)

    transformed_pos = (res.T[:, 0:3]).reshape(original_shape)

    return transformed_pos


# ---------------------------------------
'''
    input: 
        velocities: 
            shape [-1, 6]
        target_frame_relative_quat: 
            target frame's quaternion in current frame [x, y, z, w]
    output:
        transformed_velocities: 
            shape [-1, 6]
'''
def transformVelocities(velocities, 
                        target_frame_relative_quat=None,
                        target_frame_relative_quat_inv=None):
    
    if (target_frame_relative_quat is None) and (target_frame_relative_quat_inv is None):
        raise NameError("Both target_frame_relative_quat and target_frame_relative_quat_inv are None !")
    elif (target_frame_relative_quat is not None) and (target_frame_relative_quat_inv is not None):
        raise NameError("Both target_frame_relative_quat and target_frame_relative_quat_inv are not None !")
    
    velocities = np.array(velocities)
    original_shape = velocities.shape
    try:
        velocities = velocities.reshape(-1, 6)
    except:
        raise NameError("transformVelocities(): invalid input.")

    if target_frame_relative_quat is not None:
        rot_matrix = sciR.from_quat(target_frame_relative_quat).inv().as_matrix()
    elif target_frame_relative_quat_inv is not None:
        rot_matrix = sciR.from_quat(target_frame_relative_quat_inv).as_matrix()

    rot_operator = np.block([[rot_matrix, np.zeros((3,3))], 
                                    [np.zeros((3,3)), rot_matrix]])
    
    transformed_velocities = (rot_operator @ velocities.T).T
    return transformed_velocities.reshape(original_shape)


# ---------------------------------------
'''
    input:
        quat: [w, x, y, z]
'''
def quatInv(quat):
    quat = np.array(quat)
    quat_inv = quat.copy()
    quat_inv[1:] = -quat[1:]
    return quat_inv

# ---------------------------------------
'''
    Function: 
        calculate J(q1) = d(q1 * q2) / dq2
    Input:
        q1: [w, x, y, z]
    Output:
        The derivative
'''
def partialQuatMultiply(quat):
    w, x, y, z = quat

    J = np.array([[w, -x, -y, -z],
                  [x, w, -z, y],
                  [y, z, w, -x],
                  [z, -y, x, w]])
    return J


# ---------------------------------------
'''
    Function:
        calculate M, where dq/dt = M(q) * avel
    Input:
        q: [w, x, y, z]
'''
def mappingFromAvelToDquat(quat):
    w, x, y, z = quat

    M = 1.0/2.0 * np.array([[-x, -y, -z],
                            [w, -z, y],
                            [z, w, -x],
                            [-y, x, w]])
    return M






# ---------------------------------------------------------------------------------
if __name__ == '__main__':
    positions = [[0, 1, 2], [7, 4, 2]]
    target_frame_pos = [3, 4, 5]
    target_frame_quat = [0, 0, 0, 1]

    transformPositions(positions, posQuat2Isometry3d(target_frame_pos, target_frame_quat))