import numpy as np
import cv2
import matplotlib.pyplot as plt
from ..robot_camera_calibration._calibration_data_utils import tf_from_rvectvec
from ..logger import logger
from ..tags_detection._fiducial import TagDetection, Fiducial
from ..inference._grasp_predictor import GraspPredictor
from ..inference._object_pose_estimator import ObjectPoseEstimator
def transform_2_object_frame(cam2base_tf, obj_rvec_in_cam, obj_tvec_in_cam, ee_rvec_in_base, ee_tvec_in_base):
    """! Transforms rigid body frame defined in the robot base frame to the "object"s frame

    @param    cam2base_tf (numpy array): 4x4 Transformation matrix of the camera frame defined in the robot's baseframe usually gotten from robot camera calibration routine. 
    @param     obj_rvec_in_cam (numpy array): 3x1 angle-axis rotation vector of the object's pose in the camera frame. 
    @param     obj_tvec_in_cam (numpy array): 3x1 translation vector of the object's pose in the camera frame. 
    @param     ee_rvec_in_base (numpy array): 3x1 angle-axis rotation vector of the robot end-effector frame/pose defined in the robot's base frame. 
    @param     ee_tvec_in_base (numpy array): 3x1 translation vector of the robot end-effector frame/pose defined in the robot's base frame.

    @return    numpy array: 4x4 Transformation matrix of the end-effector pose in the object's frame of reference. 
    """
    obj2cam_tf = tf_from_rvectvec(obj_rvec_in_cam, obj_tvec_in_cam)
    grasp2base_tf = tf_from_rvectvec(ee_rvec_in_base, ee_tvec_in_base)
    grasp2cam_tf = np.matmul(np.linalg.inv(cam2base_tf), grasp2base_tf)
    grasp2obj_tf = np.matmul(np.linalg.inv(obj2cam_tf), grasp2cam_tf)
    return grasp2obj_tf

def plot_image(image, blocking = False, transient = True, transient_time = 3.5):
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

class Object:
    """! Class defined for an object that we want to grasp either using fiducial marker based pose estimation, or Neural Network based pose estimation or Neural Network based direct grasp pose prediction. 
    """
    def __init__(self, object_name,
                        tag:Fiducial=None,
                        grasp_predictor:GraspPredictor=None,
                        pose_estimator:ObjectPoseEstimator=None):
        """! Object Class Constructor

        @param    object_name (str): Name of the object to be grasped.
        @param    tag (Fiducial, optional): A Fiducial marker, if a fiducial marker is used to ID the object and predict grasp poses using the same. Defaults to None.
        @param    grasp_predictor (GraspPredictor, optional): GraspPredictor object that can be used to directly infer desired grasp poses for the object given an RGB(D) observation. Defaults to None.
        @param    pose_estimator (ObjectPoseEstimator, optional): ObjectPoseEstimator object that can be used estimate pose of the object. Defaults to None.
        """
        self.object_name = object_name
        self.tag = tag
        self.pose_estimator = pose_estimator
        self.grasp_predictor = grasp_predictor
        assert(not(tag is None and pose_estimator is None and grasp_predictor is None)), "You need to provide a pose estimator or grasp predictor if an AR tag is not attached to the object."

    def get_object_pose(self, color_image, camera, debug_image = True, use_tag_to_find_pose = False):
        """! Computes the object's using an RGB(D) camera sensor

        @param    color_image (numpy array): 3 channel RGB image matrix.
        @param    camera (Camera): Camera object.
        @param    debug_image (bool, optional): Should we display an image overlayed with the predicted pose for debugging? Defaults to True.
        @param    use_tag_to_find_pose (bool, optional): Is there a fiducial tag attached to the object to predict its pose? Defaults to False.

        @return    numpy array: 3x1 angle-axis rotation vector of the object's pose.
        @return    numpy array: 3x1 translation vector of the object's pose.
        @return    TagDetection or MegaPose6D Detection output: Depending on how this object class was instantiated. 
        """
        if self.pose_estimator is None or use_tag_to_find_pose: 
            return self.get_tag_pose(color_image = color_image, 
                                camera = camera, 
                                debug_image = debug_image)
        return self.pose_estimator.estimate_pose(camera, debug_image)# if using a pose estimator 

    def get_tag_pose(self, color_image, camera, debug_image = False):
        """! Gets the pose of the fiducial tag attached to the object, if a tag exists on the object. 

        @param    color_image (numpy array): 3 channel RGB image matrix.
        @param    camera (Camera): Camera object.
        @param    debug_image (bool, optional): Should we display an image overlayed with the predicted pose for debugging? Defaults to False. 
        
        @return    numpy array: 3x1 angle-axis rotation vector of the object's pose
        @return    numpy array: 3x1 translation vector of the object's pose
        @return    TagDetection: A Fiducial TagDetection object that stores 2D fiducial corner's locations and reprojection error information. 
        """
        corners, ids, _ = self.tag.detect_markers(color_image)
        if ids is None:
            logger.debug("No tag markers found in this image")
            return None,  None, None
        image_gray =  cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        self.tag.refine_corners(image_gray, corners)
        if debug_image:
            self.tag.draw_detections(color_image = color_image,
                                corners = corners)

        obj_points, img_points = self.tag.get_matched_object_points(corners, ids)
        # obj_points, img_points = self.tag.get_matched_object_points(corners)
        rvec, tvec = self.tag.estimate_pose(obj_points, img_points, camera)
        reproj_error, error_variance = self.tag.compute_reprojection_error(rvec,
                                                            tvec,
                                                            obj_points,
                                                            img_points,
                                                            camera)   
        logger.info(f"Reprojection error with mean = {reproj_error}, variance = {error_variance}")
        
        if debug_image:
            self.tag.draw_detections(color_image = color_image,
                                corners = corners)
            self.tag.draw_estimated_frame(color_image = color_image,
                                        camera = camera,
                                        rvec = rvec,
                                        tvec = tvec)
            plot_image(color_image)
            
        return rvec, tvec, TagDetection(reproj_error_mean=reproj_error, 
                                        reproj_error_variance=error_variance,
                                        matched_img_pts=img_points,
                                        matched_obj_pts=obj_points)                                                      
        
    # def get_object_pose(self, rvec_tag2world, tvec_tag2world):
    #     tf_tag2world = tf_from_rvectvec(rvec_tag2world, tvec_tag2world)
    #     return rvectvec_from_tf(np.matmul(tf_tag2world, self.object2tag_offset))

    # def compute_object_pose(self, color_image, camera, debug_image = False):
    #     rvec_tag2cam, tvec_tag2cam, tag_detection = self.get_tag_pose(color_image, camera, debug_image = debug_image)
    #     rvec_object2cam, tvec_object2cam = self.get_object_pose(rvec_tag2cam, tvec_tag2cam)
        
    #     if debug_image:
    #         self.tag.draw_estimated_frame(color_image = color_image,
    #                                     camera = camera,
    #                                     rvec = rvec_object2cam,
    #                                     tvec = tvec_object2cam)
    #         plt.imshow(color_image)
    #         plt.show(block=False)
    #         plt.pause(2.5)
    #         plt.close()   

    #     return rvec_object2cam, tvec_object2cam, tag_detection        