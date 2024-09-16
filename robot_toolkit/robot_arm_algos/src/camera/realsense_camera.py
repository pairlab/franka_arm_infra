import numpy as np
import open3d as o3d
from .realsense_driver import RealSenseDriver
from .camera import RGBDCamera
from ..logger import logger
"""
        Raises:
            Exception: If unable to detect/start the realsense camera driver
"""
class RealSenseCamera(RGBDCamera):
    """! RealSense Camera Class built using RGBDCamera Class Template
    """     
    def __init__(self, camera_matrix = None, dist_coeffs = None, start_camera = True, camera_id = None): 
        """! RealSense Camera Class constructor
        @param camera_matrix (numpy array, optional): 3x3 camera matrix. Defaults to None.
        @param dist_coeffs (numpy array, optional): 1 dimensional array of distortion parameters. Defaults to None.
        @param start_camera (bool, optional): Boolean variable that powers on the Camera if set to true. Defaults to True.
        @param camera_id (int, optional): ID of the camera. useful for realsense cameras and for multiple cameras setup. Defaults to None.
        
        @exception Exception If unable to detect/start the realsense camera driver
        """
        try:
            if start_camera:
                if camera_id is None:
                    camera_id = RealSenseDriver.get_all_device_serials()[0]
                self.realsense_driver = RealSenseDriver(serial_number = camera_id)
                self.realsense_driver.start() 
        except Exception as e:
            logger.error(e)
            raise Exception(f"Failed to start intel RealSense")

        if camera_matrix is None and dist_coeffs is None:
            intr = self.realsense_driver.get_intrinsics()            
            camera_matrix = np.array([[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy],[0.0,0.0,1.0]])
            dist_coeffs = np.array([0.0, 0.0, 0.0, 0.0])

        self.start_camera = start_camera
        RGBDCamera.__init__(self, camera_matrix,
                                dist_coeffs,
                                camera_id) 
                                  
    def get_current_rgb_frame(self,):
        """! Abstract function implementation for RealSense Camera Class. Gets curent RGB image
        
        @return numpy array: 3 channel RGB image
        """          
        color_im, _ = self.realsense_driver.read()  
        return color_im
    
    def get_current_depth_frame(self,):
        """! Abstract function implementation for RealSense Camera Class. Gets current Depth image
        
        @return numpy array: 1 channel Depth image
        """          
        _, depth_im = self.realsense_driver.read()    
        return depth_im
    
    def get_current_rgbd_frames(self,):
        """! Abstract function implementation for RealSense Camera Class. Gets both current RGB & Depth image
        
        @return numpy array: 3 channel RGB image        
        @return numpy array: 1 channel Depth image
        """         
        color_im, depth_im = self.realsense_driver.read()    
        return color_im, depth_im
    
    def __del__(self):
        """! RealSense Class Destructor. Powers Off the RealSense Camera
        """
        if self.start_camera:
            self.realsense_driver.stop()

