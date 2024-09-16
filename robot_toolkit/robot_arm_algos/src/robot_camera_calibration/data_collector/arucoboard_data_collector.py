import numpy as np
import matplotlib.pyplot as plt
import cv2
from ...tags_detection._fiducial import TagDetection
from ._data_collector import CameraDataCollector
from ...tags_detection.aruco_board import ArucoBoard, ArucoBoardData
from ...camera.camera import RGBCamera
from ...logger import logger

class ArucoBoardDataCollector(ArucoBoard, CameraDataCollector):
    """! If DataCollector class for aruco board patterns used for robot camera calibration
    """
    def __init__(self, aruco_board_data, debug_detection_image = True):
        """! ArucoBoardDataCollector constructor
        
        @param    aruco_board_data (ArucoBoardData): Data used to describe the arucoboard pattern 
        @param    debug_detection_image (bool, optional): If true, a window with the image and aruco detections and pose estimation of the tag is plotted. Defaults to True.
        """
        ArucoBoard.__init__(self, aruco_board_data)
        CameraDataCollector.__init__(self, self.fiducial_type)
        self.debug_detection_image = debug_detection_image
    
    def get_tag_frame(self, color_image, camera:RGBCamera):
        """! Detects arucotags in the arucoboard and also estimates the pose of the arucoboard calibration pattern

        @param    color_image (numpy matrix): 3 channel numpy matrix of the RGB image 
        @param    camera (RGBCamera): Camera object

        @return    numpy array: 3x1 angle-axis rotation of the calibration tag's pose
        @return    numpy array: 3x1 translation vector of the calibration tag's pose
        @return    TagDetection: TagDetection object that stores reprojection error mean, variance, cornerpoints and their corresponding world points in the arucoboard frame for all the arucotags in the board.  
        """
        corners, ids, _  = self.detect_markers(color_image)
        if ids is None:
            return None, None, None
        image_gray =  cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        self.refine_corners(image_gray, corners)
        obj_points, img_points = self.get_matched_object_points(corners, ids)
        rvec, tvec = self.estimate_pose(obj_points, img_points, camera)
        reproj_error, error_variance = self.compute_reprojection_error(rvec,
                                                            tvec,
                                                            obj_points,
                                                            img_points,
                                                            camera)     
     
        if self.debug_detection_image:
            color_image_ = color_image.copy() 
            self.draw_detections(color_image = color_image_,
                                corners = corners)
            self.draw_estimated_frame(color_image = color_image_,
                                        camera = camera,
                                        rvec = rvec,
                                        tvec = tvec)
            plt.imshow(color_image_)
            plt.show(block=False)
            plt.pause(2.5)
            plt.close()

        return rvec, tvec, TagDetection(reproj_error_mean=reproj_error, 
                                        reproj_error_variance=error_variance,
                                        matched_img_pts=img_points,
                                        matched_obj_pts=obj_points)

        