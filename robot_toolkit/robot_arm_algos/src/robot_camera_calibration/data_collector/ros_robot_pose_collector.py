import cv2
import rospy
from tf import TransformListener
import tf.transformations as tf_utils
from ._data_collector import RobotPoseCollector
from ...logger import logger

class ROSRobotPoseCollector(RobotPoseCollector):
    """! RobotPoseCollector Class Abstract implementation based on ROS TF. This class directly queries the TF tree to get the robot end-effector's pose
    """
    def __init__(self, ee_frame, base_frame, init_node = True):
        """! RobotPoseCollector Constructor

        @param    ee_frame (str): name of the robot's end-effector frame in the ROS TF tree
        @param    base_frame (str): name of the robot's base frame in the ROS TF tree
        @param    init_node (bool, optional): If true, rospy.init_node is called, needs to be false if this class is part of already initialized ROS node. Defaults to True.
        """
        if(init_node):
            try:
                rospy.init_node("tf_listener")
            except rospy.exceptions.ROSException as e:
                print("Node has already been initialized, not doing a new one for tf listener")

        self.tf_listener = TransformListener()
        self.ee_frame = ee_frame
        self.base_frame = base_frame

    
    def get_ee_frame(self,):
        """! Returns the robot's end-effector pose measured in the robot's base frame 

        @return    numpy array: 3x1 angle-axis representation of rotation
        @return    numpy array: 3x1 translation vector
        """
        t = self.tf_listener.getLatestCommonTime(self.base_frame, self.ee_frame)
        ee_position, ee_quaternion = self.tf_listener.lookupTransform(self.base_frame, self.ee_frame, t)
        ee_rotation_matrix = tf_utils.quaternion_matrix(ee_quaternion)
        ee_rotation_matrix = ee_rotation_matrix[0:3, 0:3]
        return cv2.Rodrigues(ee_rotation_matrix)[0], ee_position


    