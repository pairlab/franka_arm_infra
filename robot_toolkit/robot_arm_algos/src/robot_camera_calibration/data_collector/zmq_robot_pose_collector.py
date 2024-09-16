import numpy as np
import cv2
import zmq
from ._data_collector import RobotPoseCollector

from wrapt_timeout_decorator import *

class ZmqRobotPoseCollector(RobotPoseCollector):
    """! RobotPoseCollector Class Abstract implementation based on zmq server. This class expects a zmq server to run on the realtime computer connected to the robot. 
    Useful to move Franka manually when status LED is white. ROS TF tree maybe corrupted when status LED is turned white when using frankapy.

    """
    def __init__(self, zmq_ip, zmq_port):
        """! ZmqRobotPoseCollector Class Constructor. Sets up a ZMQ client

        @param    zmq_ip (str): IP address of the computer in which the zmq server is running
        @param    zmq_port (str): arbitrary zmq port. ensure it is the same port on the zmq server running on the realtime computer
        """
        RobotPoseCollector.__init__(self)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        socket_address = "tcp://" + zmq_ip + ":" + zmq_port 
        self.socket.connect(socket_address)
    
    @timeout(5.0)
    def get_ee_frame(self,):
        """! Returns the robot's end-effector pose measured in the robot's base frame 

        @return    numpy array: 3x1 angle-axis representation of rotation
        @return    numpy array: 3x1 translation vector
        """
        self.socket.send_string("data")#even an empty message would do
        message = self.socket.recv()# receives EE pose 
        zmq_pose = np.frombuffer(message).astype(np.float32)
        zmq_pose = np.reshape(a=zmq_pose, newshape=(4,4), order='F')
        zmq_position = np.array(zmq_pose[:3,3])
        zmq_rot = np.array(zmq_pose[:3,:3])
        return cv2.Rodrigues(zmq_rot)[0], zmq_position