from abc import ABC, abstractmethod
import time
from dataclasses import dataclass, field
from ...camera.camera import RGBCamera
from ...logger import logger
# from .._calibration_data_utils import write_data_for_fg_optim


@dataclass
class CalibrationData:
    """! Data Class to store lists of end-effector and calibration tag angle-axis rotations and translation vectors
    """
    rvecs_ee2base: list() = field(default_factory=lambda: [])
    tvecs_ee2base: list() = field(default_factory=lambda: [])
    rvecs_tag2cam: list() = field(default_factory=lambda: [])
    tvecs_tag2cam: list() = field(default_factory=lambda: [])
    
    
@dataclass
class CalibrationSolverData:
    """! Data Class to store A,B in the calibration solver equation AX = XB. 
    It is stored as lists of both tuple of (rvec,tvec) and a 4x4 numpy matrix
    """
    As: list() = field(default_factory=lambda: [])
    As_tf: list() = field(default_factory=lambda: [])
    Bs: list() = field(default_factory=lambda: [])
    Bs_tf: list() = field(default_factory=lambda: [])



class RobotPoseCollector(ABC):
    """! Abstract Class of robot end-effector pose collector
    """
    def __init__(self):
        """Abstract Class RobotPoseCollector Constructor. Does Nothing. 
        """
        pass
    @abstractmethod
    def get_ee_frame(self,):
        """! Abstract function that returns 4x4 numpy matrix of end-effector frame defined in robot base frame
        """
        pass    

class CameraDataCollector(ABC):
    """! Abstract Class of collector of Calibration tag pose defined in camera frame.
    """
    def __init__(self, calibration_tag_type):
        """! Abstract Class CameraDataCollector constructor. 

        @param    calibration_tag_type (str): Calibration tag type eg. arucotag, arcuoboard, etc 
        """
        self.calibration_tag_type = calibration_tag_type
        pass
   
    @abstractmethod
    def get_tag_frame(self, color_image, camera:RGBCamera):
        """! Abstract method that computes and returns calibration tag's pose if present in the color_image

        @param    color_image (numpy array): Input color image in calibration tag is to be detected and its pose estimated
        @param    camera (RGBCamera): camera object that has the intrinsic information as data member
        """
        pass   

class RobotCameraCalibrationDataCollector():
    """! Class used to collect robot pose data and calibration tag pose data to setup the calibration optimization problem
    """
    def __init__(self, robot_pose_collector:RobotPoseCollector,
                       tag_pose_collector:CameraDataCollector,
                        camera,
                        n_data, 
                        fg_data_folder_path = None,
                        collect_data_for_fg_optim = True,
                        robot_arm_object = None,
                        reproj_error_thresh=1.5,
                        move_robot_automatically=False):
        """! RobotCameraCalibrationDataCollector Constructor

        @param    robot_pose_collector (RobotPoseCollector): robot pose collector object
        @param    tag_pose_collector (CameraDataCollector): calibration tag pose collector object
        @param    camera (Camera): Camera object that can get current image frames and stores instrinsics in its datamembers 
        @param    n_data (int): number of poses to be collected for calibration
        @param    fg_data_folder_path (str, optional): Storing pose data in the format used by Factor Graph/reprojection error based optimization for calibration
        @param    collect_data_for_fg_optim (bool, optional): Should we also collect pose data in the format needed for Factor Graph optimization based method. Defaults to True.
        @param    robot_arm_object (RobotArm, optional): Optional robot_arm object used to move robot if automated data collection is set as True. Defaults to None.
        @param    reproj_error_thresh (float, optional): Threshold of acceptable reprojection error in the calibration tag pose estimation. measured in pixels. Defaults to 1.5.
        @param    move_robot_automatically (bool, optional): Is the robot moved automatically to collect pose data. if false, the robot arm is expected to be moved manually. Defaults to False.
        """
        self.check_robot_arm_object(robot_arm_object, move_robot_automatically)
        self.robot_arm_object = robot_arm_object
        self.robot_pose_collector = robot_pose_collector
        self.tag_pose_collector = tag_pose_collector
        self.move_robot_automatically = move_robot_automatically
        self.n_data = n_data
        self.data_idx = 0
        self.calib_data = CalibrationData()
        self.camera = camera
        self.reproj_error_thresh = reproj_error_thresh        
        self.collect_data_for_fg_optim = collect_data_for_fg_optim
        self.fg_data_folder_path = fg_data_folder_path

    def collect_data(self,):
        """! Function that encapsulates the pose data collection process 

        @exception    Exception: If something goes wrong when trying to move the robot automatically. something wrong in move_and_collect_data()
        """
        if self.move_robot_automatically:
            try:
                self.move_and_collect_data()
                return
            except Exception as error:
                logger.error(f"Something went wrong while trying to autonmously collect calibration data")
                logger.error(error)
                return

        while(True):
            logger.info(f"Press Enter to continue collecting current sample....else space bar to stop")

            ip = input()
            if (ip == ""):
                self.get_pose_measurements()
            else:
                logger.info(f"stopping data collection")
                break

            if self.data_idx > self.n_data:
                break

    def get_pose_measurements(self,):
        """! Function that gets robot pose data and calibration tag data and stores them in a list to be used for calibration
        """
        color_image = self.camera.get_current_rgb_frame()
        fg_color_image = color_image.copy()

        rvec_tag2cam, tvec_tag2cam, tag_det_params = self.tag_pose_collector.get_tag_frame(color_image,
                                                                                    self.camera)
        if rvec_tag2cam is None and tvec_tag2cam is None:
            logger.info("No markers detected in this image frame")
            return

        if tag_det_params.reproj_error_mean > self.reproj_error_thresh:
            logger.debug(f"High Reprojection error of {tag_det_params.reproj_error_mean}, variance {tag_det_params.reproj_error_variance}")
            return

        logger.debug(f"Reprojection error with mean = {tag_det_params.reproj_error_mean}, variance = {tag_det_params.reproj_error_variance}")
        rvec_ee2base, tvec_ee2base = self.robot_pose_collector.get_ee_frame()
        self.add_robot_tag_pose_pair(rvec_ee=rvec_ee2base,
                                    tvec_ee=tvec_ee2base,
                                    rvec_tag=rvec_tag2cam,
                                    tvec_tag=tvec_tag2cam)

        if (self.collect_data_for_fg_optim):
            write_data_for_fg_optim(ee_rot = rvec_ee2base,
                                    ee_pos = tvec_ee2base,
                                    color_image = fg_color_image,
                                    data_idx = self.data_idx,
                                    obj_points = tag_det_params.matched_obj_pts,
                                    img_points = tag_det_params.matched_img_pts,
                                    fg_data_folder_path = self.fg_data_folder_path)



    def add_robot_tag_pose_pair(self, rvec_tag, tvec_tag, rvec_ee, tvec_ee):
        """! Add current robot pose data and calibration tag pose data pair

        @param    rvec_tag (numpy array 3x1): angle-axis representation of rotation of the calibration tag's pose
        @param    tvec_tag (numpy array 3x1): translation vector of the calibration tag's pose
        @param    rvec_ee (numpy array 3x1): angle-axis representation of rotation of the robot end-effector's pose
        @param    tvec_ee (numpy array 3x1): translation vector of the robot end-effector's pose
        """
        self.calib_data.rvecs_ee2base.append(rvec_ee)
        self.calib_data.tvecs_ee2base.append(tvec_ee)
        self.calib_data.rvecs_tag2cam.append(rvec_tag)
        self.calib_data.tvecs_tag2cam.append(tvec_tag)
        logger.info(f"data collected so far: {self.data_idx+1}/{self.n_data}")
        # logger.info(f"{rvec_ee}, {tvec_ee}, {rvec_tag}, {tvec_tag}")
        self.data_idx += 1        


    def check_robot_arm_object(self, robot_arm_object, move_robot_automatically):
        """! Checks if robot_arm_object is given if user expects the robot to move automatically and collect calibration data

        @param    robot_arm_object (RobotArm): RobotArm object that can do position control of the robot in joint and end-effector space.
        @param    move_robot_automatically (bool): Boolean variable that says if the robot is expected to move automatically or not. 
        """
        if move_robot_automatically and robot_arm_object is None:
            logger.error(" You must pass a robot arm object if you want to collect data automatically")
            return
        else:
            logger.info(f"Make sure you point the robot to a good initial configuration looking at the calibration tag")        
        return
        
    def move_and_collect_data(self,):
        """! This function moves the robot to pesudo random poses and collects pose data for calibration
        """
        logger.info(f"Press Enter if you are happy with the initial configuration and ready to start collecting data autonomously....else space bar and Enter to stop")
        ip = input()    
        if(ip != ""):
            return 

        ee_poses = self.robot_arm_object.get_randomized_absolute_poses(n_poses = self.n_data)
        initial_pose = ee_poses[0]
        for ee_pose in ee_poses:
            self.robot_arm_object.go_to_ee_pose(ee_pose)
            time.sleep(2.0)# wait for any vibrations to subdue
            self.get_pose_measurements()
            self.robot_arm_object.go_to_ee_pose(initial_pose)
        
        return

    def get_relative_pose_between_measurements(self,):
        """! This method computes relative transformation matrices between two robot or calibration tag poses. 
        Not yet implemented
        """
        pass
    

import numpy as np
import cv2
def write_data_for_fg_optim(ee_rot, ee_pos,
                            color_image,
                            data_idx,
                            obj_points,
                            img_points,
                            fg_data_folder_path):
    """! This method stores the robot and calibration tag pose data in the format needed for factor graph based reprojection
    error minimization formulation for robot camera calibration.

    @param    ee_rot (numpy array 3x3): robot end-effector frame rotation matrix 
    @param    ee_pos (numpy array 3x1): robot end-effector frame translation vector
    @param    color_image (numpy array): color image when end-effector pose was measured
    @param    data_idx (int): index of the end-effector and calibration tag pose pair
    @param    obj_points (numpy array): 3D points of the corners defined in the calibration tag frame
    @param    img_points (numpy array): 2D keypoints of the corners in the image
    @param    fg_data_folder_path (str): path to the folder in which data is stored for factor grapph based optimization for calibration
    """

    if fg_data_folder_path is None:
        logger.error("The path to folder to store data for factor graph optimization is not set")
        return
        
    ee_pose_tf = np.eye(4)
    ee_pose_tf[0:3,-1] = ee_pos
    if np.shape(np.squeeze(ee_rot)) ==(3,3):
        ee_pose_tf[0:3,0:3] = ee_rot
    else: 
        ee_pose_tf[0:3,0:3] = cv2.Rodrigues(ee_rot)[0] 

        pose_file_name = fg_data_folder_path+"{:0>4}".format(str(data_idx))+"_pose.csv"
        kps_file_name =  fg_data_folder_path+"{:0>4}".format(str(data_idx))+"_kps.csv"
        image_file_name = fg_data_folder_path+"{:0>4}".format(str(data_idx))+"_image.jpg"
        img_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(image_file_name, img_gray)
        with open(pose_file_name, 'w+') as f:
            print(str(ee_pose_tf[0,0])+" "+str(ee_pose_tf[0,1])+" "+str(ee_pose_tf[0,2])+" "+str(ee_pose_tf[0,3]), file=f)
            print(str(ee_pose_tf[1,0])+" "+str(ee_pose_tf[1,1])+" "+str(ee_pose_tf[1,2])+" "+str(ee_pose_tf[1,3]), file=f)
            print(str(ee_pose_tf[2,0])+" "+str(ee_pose_tf[2,1])+" "+str(ee_pose_tf[2,2])+" "+str(ee_pose_tf[2,3]), file=f)
            print(str(ee_pose_tf[3,0])+" "+str(ee_pose_tf[3,1])+" "+str(ee_pose_tf[3,2])+" "+str(ee_pose_tf[3,3]), file=f)

        with open(kps_file_name, 'w+') as f:
            for(obj_pt, img_pt) in zip(obj_points, img_points):
                obj_pt = np.squeeze(obj_pt)
                img_pt = np.squeeze(img_pt)
                print(str(img_pt[0])+" "+str(img_pt[1])+", "+str(obj_pt[0])+" "+str(obj_pt[1])+" "+str(obj_pt[2]), file=f)