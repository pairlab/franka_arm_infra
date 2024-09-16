import time
import numpy as np
import threading
import cv2
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
from .camera import RGBCamera
from ..logger import logger

class ROSCamera(RGBCamera):
    """!
    A ROS Camera Node Class built using RGBDCamera Class Template
    """     
    def __init__(self, image_topic_name, camera_info_topic_name, init_node = True):
        """! ROS Camera Class Constructor. Waits/Blocks till messages are received both image and camera_info topics

        @param    image_topic_name (str): Name of the topic to subscribe images from 
        @param    camera_info_topic_name (str): Name of the topic to subscribe camera info from 
        @param    init_node (bool, optional): If true, rospy.init_node() is callled to initialize a ROS node. Not needed if this Class part of another ROS node. 
        
        @exception    Exception: rospy.exceptions.ROSException
        """
        if(init_node):
            try:
                rospy.init_node("image_subscriber", anonymous=True)
            except rospy.exceptions.ROSException as e:
                print("Node has already been initialized, not doing a new one for image susbscriber")
        self.rgb_image_msg = None
        self.camera_info_msg = None
        self.cv_bridge = CvBridge()
        rospy.Subscriber(image_topic_name, Image, self.rgb_image_callback)
        rospy.Subscriber(camera_info_topic_name, CameraInfo, self.camera_info_callback)
        while(self.rgb_image_msg is None):
            logger.info("Waiting for image msg")
            time.sleep(1)
            if (rospy.is_shutdown()):
                return
        camera_matrix, dist_coeffs = self.get_camera_params(self.camera_info_msg)
        RGBCamera.__init__(self, camera_matrix = camera_matrix,
                                dist_coeffs = dist_coeffs)

    def rgb_image_callback(self, rgb_img_msg):
        """!
        ROS Subcriber callback function for image topic
 
        @param    rgb_img_msg (sensor_msgs.msg.Image): RGB Image msg
        """       
        if self.camera_info_msg is None:
            return 
        self.rgb_image_msg = rgb_img_msg

    def camera_info_callback(self, camera_info_msg):  
        """!
        ROS Subcriber callback function for camera_info topic

        @param    camera_info_msg (sensor_msgs.msg.CameraIngo): camera info msg
        """          
        if(self.camera_info_msg is not None):
            return
        self.camera_info_msg = camera_info_msg        

    def get_current_rgb_frame(self,):
        """! Abstract function implementation for RealSense Camera Class. Gets curent RGB image

        @param    numpy array: 3 channel RGB image
        """            
        color_image = self.cv_bridge.imgmsg_to_cv2(self.rgb_image_msg)
        return color_image

    def get_camera_params(self, camera_info_msg):
        """! Gets camera matrix and distortion coeffecients

        @return    numpy array(3x3): camera matrix 
        @return    numpy array(1x4): distortion coeffecients
        """            
        dist_coeffs = np.array(camera_info_msg.D[:4])
        camera_matrix = np.array(camera_info_msg.K).reshape(3,3)
        return camera_matrix, dist_coeffs
    
    