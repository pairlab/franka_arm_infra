from abc import ABC, abstractmethod
class RobotArm(ABC):
    """! Abstract Class for a Robot Arm. Needs to be implemented for a particular type of arm. 
    """
    def __init__(self, robot_id = 0):
        """! Robot Arm Class Constructor

        @param    robot_id (int, optional): ID for Robot incase more than one is used. Defaults to 0.
        """
        self.robot_id = robot_id
    
    @abstractmethod    
    def get_ee_frame(self,):
        """! Abstract method that returns a 4x4 Transformation matrix of the End-Effector frame in the Base Frame
        """
        pass

    @abstractmethod
    def get_joint_angles(self,):
        """! Abstract method that returns the robot arms current joint angles 
        """
        pass

    @abstractmethod
    def go_to_joint_angles(self, goal_joint_config):
        """! Abstract method that moves the robot arm to a desired joint angle configuration


        @param    goal_joint_config: desired joint angle configuration 
        """
        pass

    @abstractmethod
    def go_to_ee_pose(self, goal_ee_pose):
        """! Abstract method that moves the end-effector to a desired pose defined in the base frame

        @param    goal_ee_pose (numpy array): 4x4 Matrix of the desired End-Effector frame in the robot base frame
        """
        pass

    @abstractmethod
    def go_to_ee_pose_delta(self, delta_ee_pose):
        """! Abstract method that moves the end-effector to a desired delta pose defined in the base frame

        @param    goal_ee_pose (numpy array): 4x4 Matrix of the desired End-Effector frame in the robot base frame
        """        
        pass

    # @abstractmethod 
    # def open_gripper(self,):
    #     pass

    # @abstractmethod
    # def close_gripper(self,):
    #     pass

    # @abstractmethod 
    # def close_gripper(self,):
    #     pass

