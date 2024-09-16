import numpy as np
from scipy.spatial.transform import Rotation as R
import cv2
from .data_collector._data_collector import CalibrationSolverData, CalibrationData
# from ._calibration_solver import solvers
from ..logger import logger
from ..config_reader import read_yaml_file


solvers = {"HAND_EYE_ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
            "HAND_EYE_DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS,
            "HAND_EYE_HORAUD" : cv2.CALIB_HAND_EYE_HORAUD,  
            "HAND_EYE_PARK" : cv2.CALIB_HAND_EYE_PARK,
            "HAND_EYE_TSAI" : cv2.CALIB_HAND_EYE_TSAI}     

def tf_from_rvectvec(rvec, tvec):
    """! Gets 4x4 Transformation matrix for 3x1 angle-axis rotation and 3x1 translation

    @param:    rvec (numpy array): 3x1 angle-axis rotation vector
    @param:    tvec (numpy array): 3x1 translation vector

    @return    numpy array: 4x4 Transformation matrix
    """
    out = np.eye(4)
    out[0:3, -1] = np.squeeze(tvec)
    if np.shape(rvec) == (3,1):
        out[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
    else: 
        out[0:3, 0:3] = rvec
    return out

def rvectvec_from_tf(tf):
    """! Gets 3x1 angle-axis rotation and 3x1 translation from 4x4 Transformation matrix

    @param    tf (numpy array): 4x4 Transformation matrix

    @return    numpy array: 3x1 angle-axis rotation vector
    @return    numpy array: 3x1 translation vector
    """
    rvec = cv2.Rodrigues(tf[0:3,0:3])[0]
    tvec = tf[0:3, -1]
    return rvec, tvec

def error_rot_trans(tf1, tf2):
    """Computes Difference between two transformation matrices

    @param    tf1 (numpy array): 4x4 Transformation matrix
    @param     tf2 (numpy array): 4x4 Transformation matrix

    @return    numpy array: 3x1 rotation error as euler angles in degrees
    @return    numpy array: 3x1 translation error vector in centimeters
    """
    assert(np.shape(tf1)==(4,4))
    assert(np.shape(tf2)==(4,4))
    difference = np.matmul(np.linalg.inv(tf1), tf2)
    difference_r = R.from_matrix(difference[0:3, 0:3])
    rot_error = difference_r.as_euler('xyz', degrees=True)
    trans_error = difference[0:3,-1] * 100.0 #converts from meters to cm
    return rot_error, trans_error
    

def store_robot_cam_extrinsics_result():
    pass

def read_calibration_data_from_file(file_name:str, flag_camera_in_hand=True):
    """! Reads pose data collected and stored in a file and prepares it for the calibration solver. 

    @param     file_name (str): Path to the text file in which calibration pose data is stored
    @param     flag_camera_in_hand (bool, optional): If the camera is attached to the robot's hand or its environment. Defaults to True.

    @return    CalibrationSolverData: all the robot end-effector and calibration tag pose data collected for robot camera calibration's solver. 
    """
    calib_solver_data = CalibrationSolverData()
    with open(file_name, 'r') as fp:
        lines = fp.readlines()

    for line in lines:
        data = line.split('\n')[0].split(',')
        ee_pose = tuple(((float(data[0]),
                        float(data[1]),
                        float(data[2])), 
                        (float(data[3]),
                        float(data[4]),
                        float(data[5]))))
        tag_pose = tuple(((float(data[6]),
                        float(data[7]),
                        float(data[8])), 
                        (float(data[9]),
                        float(data[10]),
                        float(data[11]))))    
        ee_pose_tf = np.eye(4); tag_pose_tf = np.eye(4)
        ee_pose_tf[0:3, 0:3] = cv2.Rodrigues(ee_pose[0])[0]
        ee_pose_tf[0:3, -1] = ee_pose[1]
        tag_pose_tf[0:3, 0:3] = cv2.Rodrigues(tag_pose[0])[0]
        tag_pose_tf[0:3, -1] = tag_pose[1] 

        calib_solver_data.Bs.append(tag_pose)
        calib_solver_data.Bs_tf.append(tag_pose_tf)

        if(flag_camera_in_hand):
            calib_solver_data.As.append(ee_pose)
            calib_solver_data.As_tf.append(ee_pose_tf)
            continue

        ee_pose_tf_inv = np.linalg.inv(ee_pose_tf)
        ee_pose_inv = list(ee_pose)
        ee_pose_inv[0] = cv2.Rodrigues(ee_pose_tf_inv[0:3, 0:3])[0]
        ee_pose_inv[1] = ee_pose_tf_inv[0:3, -1]
        ee_pose_inv = tuple(ee_pose_inv)

        calib_solver_data.As.append(ee_pose_inv)
        calib_solver_data.As_tf.append(ee_pose_tf_inv)
    
    return calib_solver_data


def write_calibration_data_file(calib_data, file_name):
    """! This function writes the robot end-effector and calibration tag pose data collected into a text file that can be used to run calibration routine later on. 

    @return    calib_data (CalibrationData): Angle-axis rotatin and translation vectors of robot-end effector and calibration tag pose pairs
    @return    file_name (str): Path of the text file to which the calibration pose data is written to 
    """
    open(file_name, 'w').close()#emptys the file

    for rvec_ee, tvec_ee, rvec_tag, tvec_tag in zip(calib_data.rvecs_ee2base,
                                                    calib_data.tvecs_ee2base,
                                                    calib_data.rvecs_tag2cam,
                                                    calib_data.tvecs_tag2cam):
        ee_pose_line = [str(i) for i in [rvec_ee[0][0], 
                                        rvec_ee[1][0], 
                                        rvec_ee[2][0], 
                                        tvec_ee[0], 
                                        tvec_ee[1], 
                                        tvec_ee[2]]]
        tag_pose_line = [str(i) for i in [rvec_tag[0][0], 
                                        rvec_tag[1][0], 
                                        rvec_tag[2][0], 
                                        tvec_tag[0][0], 
                                        tvec_tag[1][0], 
                                        tvec_tag[2][0]]]
        output_each_line = ee_pose_line + tag_pose_line
        write_to_file(output_each_line, file_name)

def read_cam_robot_extrinsics(extrinsics_file_name, method="HAND_EYE_DANIILIDIS"):
    if method not in list(solvers.keys()):
        logger.error("f{method} not prsent in the list of available solvers}")
        return
    extrinsics_dict = read_yaml_file(extrinsics_file_name)
    return extrinsics_dict["HAND_EYE_DANIILIDIS"]    
    
def write_to_file(line_list, file_name):
    """! Appends line by line to a file
    
    @param    line_list (str list): list of lines of text to be added to the text file. 
    @param    file_name (str): Path to the text file in which we want to write 
    """
    # open file in append mode
    line  = ",".join(line_list)
    with open(file_name, 'a') as f:
        f.write(line)
        f.write('\n') 
        
def transform_between_cam_robot_frames(cam2base_tf, rvec, tvec, to_camera_frame=True):
    if to_camera_frame:
        frame_in_base = tf_from_rvectvec(rvec, tvec)
        base2cam = np.linalg.inv(cam2base_tf)   
        return rvectvec_from_tf(np.matmul(base2cam, frame_in_base))
    frame_in_cam = tf_from_rvectvec(rvec, tvec)
    return rvectvec_from_tf(np.matmul(cam2base_tf, frame_in_cam))        

"""todo"""
# erb to a static frame publisher in launch file