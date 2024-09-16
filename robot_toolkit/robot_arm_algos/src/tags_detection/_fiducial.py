from dataclasses import dataclass
import numpy as np
from abc import ABC, abstractmethod
import cv2
from ..camera.camera import RGBCamera

@dataclass
class TagDetection:
    """!
    DataClass containing information pertaining to fiducial marker detection in the image. reproj_error_mean is the mean reprojection error of all the markers in a board and reproj_error_variance is its variance. 
    matched_obj_pts and matched_img_pts are the 3D locations of the corners of each marker in the tag's coordinate frame and its corresponding 2D location in the image respectively. 
    """    
    reproj_error_mean: float
    reproj_error_variance: float
    matched_obj_pts: np.ndarray
    matched_img_pts: np.ndarray

class Fiducial(ABC):
    """!
    Abstract Class that needs to be implemented for all types of fiducial marker patterns
    """
    def __init__(self, marker_type, fiducial_data):
        """!Fiducial Class Constructor 

		@param	marker_type (str): Name of the type of fiducial marker pattern
		@param fiducial_data (depends on the fiducial type): Data related to the fiducial markers
  		"""

        self.fiducial_type = marker_type
        self.fiducial_data = fiducial_data
    
    @abstractmethod
    def detect_markers(self, color_image):
        """! Detects the fiducial markers in the RGB image

		@param	color_image (numpy array): 3 channel numpy matrix of the RGB image
		"""
        pass
    
    @abstractmethod
    def estimate_pose(self, img_points, 
							obj_points, 
							camera:RGBCamera):
        """! Estimates the pose of the fiducial pattern with respect to the camera frame
        
		@param	img_points (numpy array): 2D locations of the fiducial marker's corners in the image.
		@param	obj_points (numpy array): corresponding 3D locations of the fiducial marker's corners in the fiducial marker's frame of reference.
		@param	camera (RGBCamera): Camera object that contains the intrinsics and lens distortion parameters
		"""
        pass

    @abstractmethod
    def compute_reprojection_error(self, rvec, 
										tvec, 
										obj_points, 
										img_points, 
										camera:RGBCamera):
        """! Computes reprojection error for the estimated pose of the fiducial marker using the fiducial markers detection in the image and its corresponding 3D location in the fiducial marker frame. 

		@param	rvec (numpy array): 3x1 angle axis rotation vector of the fiducial marker's pose
		@param	tvec (numpy array): 3x1 translation vector of the fiducial marker's pose
		@param	img_points (numpy array): 2D locations of the fiducial marker's corners in the image.
		@param	obj_points (numpy array): corresponding 3D locations of the fiducial marker's corners in the fiducial marker's frame of reference.
		@param	camera (RGBCamera): Camera object that contains the intrinsics and lens distortion parameters
		"""
        pass	


ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000
}

# for april tag
# https://pyimagesearch.com/2020/12/28/determining-aruco-marker-type-with-opencv-and-python/
