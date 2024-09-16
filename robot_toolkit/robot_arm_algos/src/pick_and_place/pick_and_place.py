"""
Pick n Place example with one robot and one camera with known robot camera extrinsics.

Author: Ruthrash Hari
Date: 2/10/23 
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dataclasses
from ..robot_camera_calibration._calibration_data_utils import tf_from_rvectvec, rvectvec_from_tf
from ..logger import logger
from ..config_reader import read_yaml_file

def get_offset_from_pose(pose, offset_dist = 0.15, offset_in_obj_frame = True):
    """!
    Computes a pose offset by a distance in the -ve Z direction in either the robot's base frame or in the object's frame. Useful to compute pregrasp, post place poses in pick and place applications. 
    
    @param    pose (numpy array): 4x4 Transformation matrix representation of a 6D pose. 
    @param    offset_dist (float, optional): Distance of the offset in -ve Z direction. Defaults to 0.15.
    @param    offset_in_obj_frame (bool, optional): If true, offset is computed in the -ve Z direction in the object frame, if false, in the robot's base frame. Defaults to True.

    @return    numpy array: 4x4 Transformation matrix representation of a 6D pose offset from the input `pose`
    """
    offsetvect_poseframe = np.array([0.0, 0.0, offset_dist])
    offset_pose = pose.copy()  
    if offset_in_obj_frame:
        grasp2robotbase_rot = offset_pose[0:3, 0:3]
        offsetvect_robotbaseframe = np.matmul(grasp2robotbase_rot, -1.0 * offsetvect_poseframe)
    else:
        offsetvect_robotbaseframe = offsetvect_poseframe
    offset_pose[0:3,-1] += offsetvect_robotbaseframe
    return offset_pose

def plot_image(image, blocking = True, transient = False, transient_time = 3.5):
    """! Displaying an image on a GUI

    @param     image (numpy array): 3 channel RGB or 1 channel grayscale image matrix. 
    @param     blocking (bool, optional): If True, the image GUI window will block execution. Defaults to False.
    @param     transient (bool, optional): If True, the image GUI window will be rendered in the screen for a fixed amount of time. Defaults to True.
    @param     transient_time (float, optional): The time for which GUI window will be on the screen blocking further execution of the program. Defaults to 3.5.
    """
    plt.imshow(image)
    plt.show(block=blocking)
    if transient:
        plt.pause(transient_time)
        plt.close() 
        
def draw_estimated_frame(color_image, camera, rvec, tvec):
    """! Plots the given pose as a rigidbody frame in the given image. 

    @param     color_image (numpy array): 3 channel RGB or 1 channel grayscale image matrix.
    @param     camera (Camera): an object of the Camera class
    @param     rvec (numpy array): 3x1 angle-axis rotation vector of the pose to be drawn on the image. 
    @param     tvec (numpy array): 3x1 translation vector of the pose to be to be drawn on the image. 

    @return    numpy array: Image with the given pose drawn as a rigidbody frame in the given image.
    """    
    cv2.drawFrameAxes(color_image, camera.camera_matrix, camera.dist_coeffs, rvec, tvec, 0.05)
    return color_image    

class PickAndPlace:
    """! A Class that encapsulates pick and place capbilities of a physical object using a robot arm. 
    """
    def __init__(self, robot_arm_object, camera, cam_extrinsics, camera_in_hand = True):
        """! PickAndPlace Class constructor 

        @param    robot_arm_object (RobotArm): an object of the RobotArm Class representing a real robot arm hardware.
        @param    camera (Camera): an object of the Camera class representing camera sensor hardware. 
        @param    cam_extrinsics (numpy array): 4x4 Transformation matrix of the camera frame in the robot end-effector frame (if camera_in_hand is true) or robot base frame (otherwise)
        @param    camera_in_hand (bool, optional): _description_. If the camera is attached to the robot's end-effector. False, if the camera is attached to the robot's environment. 
        """
        self.robot_arm = robot_arm_object
        self.camera = camera
        self.cam_extrinsics = cam_extrinsics#either cam2base: camera in environment or cam2gripper: camera in hand
        self.camera_in_hand = camera_in_hand 
        
    def get_cam2base_tf(self,):
        """! Gets current Camera frame defined in the robot's base frame using the robot camera extrinsic information. 

        @return    numpy array: 4x4 Transformation matrix of the camera frame in the robot's base frame. 
        """
        if(self.camera_in_hand):        
            ee2base_tf = self.robot_arm.get_ee_frame()
            return np.matmul(ee2base_tf, self.cam_extrinsics)
        else: 
            return self.cam_extrinsics
    
    def pick_object(self, object_, grasps_file, grasp_name = None):
        """! Computes a desired end-effector pose to grasp the `object_` and moves the robot arm to a pre-grasp pose, then to the grasp pose, closes gripper to grasp object and then moves to a post-grasp pose. 

        @param    object_ (Object): A predefined Object class' object. 
        @param    grasps_file (str): Path to the file where pre defined grasp poses with respect to a canonical oject frame is stored using RecordGrasps class. 
        @param    grasp_name (str, optional): Name of the grasp pre-recorded using RecordGrasps. Defaults to None.

        @return    numpy array: 4x4 Transformation matrix of the desired end-effector pose to grasp the `object_`.
        """
        logger.debug(f"performing grasps for object{object_.object_name}")
        grasp_pose = self.get_grasp_pose(object_ = object_,
                                         grasps_file = grasps_file,
                                         grasp_name = grasp_name)
        pre_grasp_pose = get_offset_from_pose(pose = grasp_pose, offset_in_obj_frame = True)
        post_grasp_pose = get_offset_from_pose(pose = grasp_pose, offset_in_obj_frame = False)
        self.robot_arm.open_gripper()
        self.robot_arm.go_to_ee_pose(pre_grasp_pose)
        self.robot_arm.go_to_ee_pose(grasp_pose)
        self.robot_arm.close_gripper()
        self.robot_arm.go_to_ee_pose(post_grasp_pose)
        return grasp_pose
    
    def place_object(self, place_pose):
        """! Places the `object_` in a desired place_pose by first moving the robot end-effector to the pre-place pose, then to the place pose, opens the gripper and then moves to a post-place pose. 
        
        @param    place_pose (numpy arra): 4x4 Transformation matrix of the desired end-effector pose to place the `object_`.
        
        @return    numpy array: 4x4 Transformation matrix of the desired end-effector pose where the `object_` was placed. 
        """
        
        # place_pose[2,3] += 0.05
        preplace_pose = get_offset_from_pose(pose = place_pose, offset_in_obj_frame = True)
        postplace_pose = get_offset_from_pose(pose = place_pose, offset_in_obj_frame = False)
        self.robot_arm.go_to_ee_pose(preplace_pose)
        self.robot_arm.go_to_ee_pose(place_pose)
        self.robot_arm.open_gripper()
        self.robot_arm.go_to_ee_pose(postplace_pose)
        return place_pose
        
    def get_grasp_pose(self, object_, grasps_file, grasp_name=None):
        """! Computes a desired end-effector pose to grasp the `object_`

        @param    object_ (Object): an object of the Object class encapsulating the physical object that we want to grasp.
        @param    grasps_file (str): path to the file containing grasp poses in the object canonical frame collected prior using the RecordGrasps Class. 
        @param    grasp_name (str, optional): name of the grap in the grasps_file defined the object's canonical frame that we wish to execute. Defaults to None.

        @return    numpy array: 4x4 Transformation matrix represention the desired end-effector pose to grasp the `object_` defined in the robot base frame. 
        """
        pre_recorded_grasps = read_yaml_file(grasps_file)
        if grasp_name is None:
            desired_grasp2obj_tf = list(pre_recorded_grasps["grasps"].values())[0]
        else: 
            desired_grasp2obj_tf = pre_recorded_grasps["grasps"][grasp_name]           

        image = self.camera.get_current_rgb_frame()
        obj2cam_rvec, obj2cam_tvec, _ =  object_.get_object_pose(color_image = image,
                                                        camera = self.camera)
        
        viz_image = draw_estimated_frame(image, self.camera, obj2cam_rvec, obj2cam_tvec)
        
        
        obj2cam_tf = tf_from_rvectvec(obj2cam_rvec, obj2cam_tvec)

        
        grasp_rvec, grasp_tvec = rvectvec_from_tf(np.matmul(obj2cam_tf, desired_grasp2obj_tf))
        viz_image = draw_estimated_frame(viz_image, self.camera, grasp_rvec,grasp_tvec)
        plot_image(viz_image)
        
        
        grasp_pose_in_base = np.matmul(self.get_cam2base_tf(),
                                    np.matmul(obj2cam_tf, desired_grasp2obj_tf))       
        return grasp_pose_in_base