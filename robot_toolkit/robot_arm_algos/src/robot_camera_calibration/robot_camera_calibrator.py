import numpy as np
from ._calibration_solver import CalibrationSolver
from ..camera.camera import RGBCamera
from .data_collector._data_collector import RobotCameraCalibrationDataCollector
from ._calibration_data_utils import write_calibration_data_file
from ..logger import logger
from ..config_reader import write_dict_to_yaml

class RobotCameraCalibrator(RobotCameraCalibrationDataCollector, CalibrationSolver):
    """! collects pose data for robot camera extrinsics calibration and outputs results from calibration problem solver
    """
    def __init__(self, robot_pose_collector, 
                tag_pose_collector,
                camera,
                n_data,
                calib_data_file_name,
                output_file,
                move_robot_automatically,                
                robot_arm_object = None,
                fg_data_folder_path = None,
                flag_camera_in_hand = True,
                collect_data_and_calibrate = True):
        """! RobotCameraCalibrator Class Constructor. Collects pose datas used for robot camera calibration.

        @param    robot_pose_collector (RobotPoseCollector): RobotPoseCollector Class object that is used to get the robot's end-effector pose in the robot base frame.
        @param    tag_pose_collector (CameraDataCollector): CameraDataCollector Class object that is used to get the calibration tag's pose in the camera frame.
        @param    camera (Camera): Camera Class object that has the camera intrinsics and functions to get the current RGB image frame.
        @param    n_data (int): Number of calibration tag and robot end-effector pose pairs to be collected for calibration.
        @param    calib_data_file_name (str): Path to the text file where robot end-effector and calibration tag's pose are written to. 
        @param    output_file (str): Path to the YAML file to which the robot camera calibration results are stored in 
        @param    move_robot_automatically (bool): Is the robot going to move automatically to sample robot end-effector and calibration tag pose pairs?
        @param    robot_arm_object (RobotArm, optional): A RobotArm Class object that is used to position control of the robot arm in the joint and end-effector space. Defaults to None.
        @param    fg_data_folder_path (str, optional): Path to the folder where measurements are stored to in the format required to run factor graph based optimization for calibration. Defaults to None.
        @param    flag_camera_in_hand (bool, optional): Is the camera attached to the robot's end-effector. false means it is attached to the robot's environment. Defaults to True.
        @param    collect_data_and_calibrate (bool, optional): If true, new pose data are collected and calibration solver is run on them. If false, calibration solver is run on previously collected pose data . Defaults to True.
         
        @exception    Exception: If < 3 robot end-effector, calibration tag pose pairs are only present in the calibration solver data. 
        """
        RobotCameraCalibrationDataCollector.__init__(self, robot_pose_collector,
                                                    tag_pose_collector,
                                                    camera,
                                                    n_data,
                                                    robot_arm_object = robot_arm_object,
                                                    move_robot_automatically = move_robot_automatically,
                                                    fg_data_folder_path = fg_data_folder_path)     
        if(collect_data_and_calibrate):
            self.collect_data()
            try:
                assert(len(self.calib_data.rvecs_ee2base) >= 3)
            except:
                logger.error(f"You need more than 3 data points, right now you only have {len(self.calib_data.rvecs_ee2base)} ")
                return
            write_calibration_data_file(self.calib_data, calib_data_file_name)
        
        CalibrationSolver.__init__(self, calib_data_file_name, flag_camera_in_hand) 
        self.output_file = output_file
    
    def run_calibration(self, solver = None):
        """! Runs Calibration solver/routine on the collected robot end-effector, calibration tag pairs of pose data

        @param    solver (str, optional): Name of the OpenCV Solver to use. If None, all solvers are run. Defaults to None.

        """
        if solver is None:
            results = self.run_calibration_all_solvers()
            file_output = dict()
            for result in results:
                file_output[result[1]] =  result[0].tolist()
            write_dict_to_yaml(file_output, self.output_file)
            return 
        
        self.run_calibration_solver(solver)


    








            

            







