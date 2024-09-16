from typing import Optional
from dataclasses import dataclass
import numpy as np
import cv2 
from ._fiducial import Fiducial, ARUCO_DICT
from ..camera.camera import RGBCamera
from ..logger import logger

@dataclass
class ArucoBoardData:
    """! DataClass for defining the ArucoBoard

    @param    dictionary (str): defines the dictionary of the aruco marker 
    @param    marker_length (float): edge length of the square aruco marker in meters
    @param    marker_separation (float): distance between two square aruco markers in the board
    @param    n_rows (int): number of rows of aruco markers in the board 
    @param    n_cols (int): number of columns of aruco markers in the board
    @param    ids(list, Optional): You can specify the ID of the markers to be rendered in the ArucoBoard.Defaults to None
    """
    dictionary: str
    marker_length: float
    marker_separation: float 
    n_rows: int
    n_cols: int
    ids: Optional[list] = None

class ArucoBoard(Fiducial):
    """! Implementation of the abstract Fiducial Class for ArucoBoard pattern
    """
    def __init__(self, aruco_board_data:ArucoBoardData):
        """! ArucoBoard Class Constructor

        @param    aruco_board_data (ArucoBoardData): ArucoBoardData data class object that defines the parameters of the arucoboard pattern
        """
        Fiducial.__init__(self, "aruco_board", aruco_board_data)
        self.create_aruco_board_detector(aruco_board_data)

    def create_aruco_board_detector(self, aruco_board_data):
        """! Instantiates aruco_board detector function to detect aruco board in the image 

        @param    aruco_board_data (ArucoBoardData): ArucoBoardData data class object that defines the parameters of the arucoboard pattern

        @exception    Exception: KeyError if the requested dictionary doesn't exist in available dictionaries of aruco markers.
        """
        try:
            dictionary = ARUCO_DICT[aruco_board_data.dictionary]
        except KeyError:
            logger.error(f"{aruco_board_data.dictionary} is unknown dictionary.")

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
        
        if aruco_board_data.ids is not None:
            logger.info("IDs:", np.array(aruco_board_data.ids))
            self.board = cv2.aruco.GridBoard((aruco_board_data.n_rows, aruco_board_data.n_cols), 
                                            aruco_board_data.marker_length, 
                                            aruco_board_data.marker_separation, 
                                            self.aruco_dict,
                                            ids = np.array(aruco_board_data.ids))
        else:
            self.board = cv2.aruco.GridBoard((aruco_board_data.n_rows, aruco_board_data.n_cols), 
                                            aruco_board_data.marker_length, 
                                            aruco_board_data.marker_separation, 
                                            self.aruco_dict)            
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, 
                                                self.aruco_params)
        self.fiducial_data = aruco_board_data
    
    def refine_corners(self, image, corners):
        """! Performs subpixel refinement of the detected corners in the image. Refined subpixel values are stored in 'corners' variable. 

        @param    image (numpy array): 1 channel grayscale image
        @param    corners (numpy array): numpy array of 4 corners for each aruco marker of the arucoboard detected in the image
        """
        winSize = [5, 5]
        zeroZone = [-1, -1]
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_COUNT, 30, 0.001)
        for corner in corners: 
            cv2.cornerSubPix(image, corner, winSize, zeroZone, criteria)

    def detect_markers(self, color_image):
        """!
        Performs aruco markers detection in the input image 

        @param    color_image (numpy array): 3 channel RGB image

        @return    numpy array: 4 corners for each aruco marker of the arucoboard detected in the image
        @return    numpy array: ids of the markers corresponding to the 'corners'
        @return    numpy array: contains the imgPoints of those squares whose inner code has not a correct codification. Useful for debugging purposes.
        """
        image_gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejectedImgPoints = self.detector.detectMarkers(image_gray)
        self.refine_corners(image_gray, corners)
        if ids is None:
            # logger.error("No corners detected in this image")
            return None, None, None
        return corners, ids, rejectedImgPoints
        
    def get_matched_object_points(self, corners, ids):
        """! Get the corresponding 3D locations of the markers' corners defined in the arucoboard's frame

        @param    corners (numpy array):  of 4 corners for each aruco marker of the arucoboard detected in the image
        @param    ids (list): list of ids corresponding to 4 corners in the corners variable

        @return    numpy array: 3D locations of the markers's corners defined in the arucoboard's frame
        @return    numpy array: corresponding 2D locations of the markers' corners in the image
        """
        obj_points = None; img_points = None
        obj_points, img_points = self.board.matchImagePoints(corners, 
                                                            ids, 
                                                            obj_points, 
                                                            img_points)
        return obj_points, img_points          

    def estimate_pose(corners, obj_points, img_points, camera:RGBCamera):
        """! Estimates pose of the arucoboard in the camera frame

        @param    corners (numpy array):  of 4 corners for each aruco marker of the arucoboard detected in the image
        @param    obj_points (numpy array): 3D locations of the markers's corners defined in the arucoboard's frame
        @param    img_points (numpy array): corresponding 2D locations of the markers' corners in the image
        @param    camera (RGBCamera): Camera Object that has the intrinsics and distortion parameters.

        @return    numpy array: 3x1 angle-axis rotation vector of the arucoboard's pose in the camera frame.
        @return    numpy array: 3x1 translation vector of the arucoboard's pose in the camera frame.
        """
        rvec = None
        tvec = None
        retval, rvec, tvec = cv2.solvePnP(obj_points, 
                                        img_points, 
                                        camera.camera_matrix, 
                                        rvec, 
                                        tvec)
        return rvec, tvec


    def compute_reprojection_error(self, rvec, tvec, obj_points, img_points, camera):
        """! Computes reprojection error mean and variance from each pair of 2D-3D correspondences

        @param    rvec (numpy array): 3x1 angle-axis rotation vector of the arucoboard's pose in the camera frame.
        @param    tvec (numpy array): 3x1 translation vector of the arucoboard's pose in the camera frame.
        @param    obj_points (numpy array): 3D locations of the markers's corners defined in the arucoboard's frame
        @param    img_points (numpy array): corresponding 2D locations of the markers' corners in the image
        @param    camera (RGBCamera): Camera Object that has the intrinsics and distortion parameters.

        @return    float: reprojection error mean from each pair of 2D-3D correspondences
        @return    float: reprojection error variance from each pair of 2D-3D correspondences
        """
        errors = []  
        for img_point, obj_point in zip(img_points, obj_points):
            proj_img_point, _ = cv2.projectPoints(obj_point,
                                                rvec,
                                                tvec,
                                                camera.camera_matrix,
                                                camera.dist_coeffs) 
            error = cv2.norm(np.squeeze(img_point), np.squeeze(proj_img_point), cv2.NORM_L2)     
            errors.append(error)
        return np.mean(errors), np.var(errors)           

    def draw_detections(self, color_image, corners):
        """! Draw detections of the markers in the color image

        @param    color_image (numpy array): 3 channel RGB image on which the detections will be drawn
        @param    corners (numpy array): 2D locations of all the corners detected in the image

        @return numpy array: 3 channel RGB image which has markers detections drawn on it.
        """
        cv2.aruco.drawDetectedMarkers(color_image, corners, borderColor=(0, 0, 255))
        return color_image
        
    def draw_estimated_frame(self, color_image, camera, rvec, tvec):
        """! Draws a frame of the arucboard's frame based on its estimated pose in the camera frame.

        @param color_image (numpy array): 3 channel RGB image on which the detections will be drawn
        @param    camera (Camera): Camera Object that has the intrinsics and distortion parameters.
        @param    rvec (numpy array): 3x1 angle-axis rotation vector of the arucoboard's pose in the camera frame.
        @param    tvec (numpy array): 3x1 translation vector of the arucoboard's pose in the camera frame.

        @return    numpy array: 3 channel RGB image which has arucboard's estimated frame drawn on it.
        """
        cv2.drawFrameAxes(color_image, camera.camera_matrix, camera.dist_coeffs, rvec, tvec, 0.1)
        return color_image

                                      