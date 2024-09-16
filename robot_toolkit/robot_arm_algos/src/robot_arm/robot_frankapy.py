import numpy as np
import random
import math
import tf.transformations as ros_tf_utils
from autolab_core import RigidTransform
from frankapy import FrankaArm
from ._robot_arm import RobotArm
from ..logger import logger
from wrapt_timeout_decorator import *

class RobotFrankaPy(RobotArm):
    """! RobotArm Abstract Class Implementation for Franka Emika Robot controlled using frankapy 
    """
    @timeout(5.0)
    def __init__(self, with_franka_gripper = True ,robot_id = 1, init_node = True):
        """! Franka Robot Arm Class Constructor

        @param    with_franka_gripper (bool, optional): True denotes that Franka Gripper is attached to the arm. Defaults to True.
        @param    robot_id (int, optional): ID used to identify which robot arm. Defaults to 1.
        @param    init_node (bool, optional): If true, rospy.init_node() is called. Could be False if this class is a part of another ROS Node. Defaults to True.
        """
        self.fpy_object = FrankaArm(robot_num = robot_id,
                                    with_gripper = with_franka_gripper,
                                    init_node = init_node)

    def get_ee_frame(self,):
        """! Abstract function implementation of RobotArm class

        @return    numy array 4x4: Transformation matrix of end-effector frame in the robot base frame
        """
        ee_frame = self.fpy_object.get_pose()
        output_ee_frame = np.eye(4)
        output_ee_frame[0:3, 0:3] = ee_frame.rotation
        output_ee_frame[0:3,-1] = ee_frame.translation
        return output_ee_frame
    
    def get_joint_angles(self,):
        """! Abstract function implementation of RobotArm class

        @return    numpy array: Franka Arms current joint angle configuration
        """
        return self.fpy_object.get_joints()
    
    def go_to_joint_angles(self, goal_joint_config, 
                                ignore_virtual_walls = True, 
                                use_impedance = False, 
                                duration = 5):
        """! Abstract function implementation of RobotArm class. Moves the robot arm to a desired joint angle configuration

        @param    goal_joint_config (numpy array): Desired joint angle configuration to go to
        @param    ignore_virtual_walls (bool, optional): Should frankapy ignore virtual walls. Defaults to True.
        @param    use_impedance (bool, optional): Should we run impedance control or position control. Defaults to False.
        @param    duration (int, optional): Frankapy needs to know the duration the joint position/impedance controller will run for in secs. Defaults to 5.
        """
        self.fpy_object.goto_joints(joints = goal_joint_config,
                                    ignore_virtual_walls = ignore_virtual_walls,
                                    use_impedance = use_impedance,
                                    duration = duration)
        return
    
    def go_to_ee_pose(self, goal_ee_pose, 
                            ignore_virtual_walls = True, 
                            use_impedance = False, 
                            duration = 3):
        """! Abstract function implementation of RobotArm Class. Moves the end-effector to a desired pose defined in the base frame

        @param    goal_ee_pose (numpy array): 4x4 Transformation Matrix of desired End-Effector pose in the Robot base frame.
        @param    ignore_virtual_walls (bool, optional): Should frankapy ignore virtual walls. Defaults to True.
        @param    use_impedance (bool, optional): Should we run impedance control or position control. Defaults to False.
        @param    duration (int, optional): Frankapy needs to know the duration the joint position/impedance controller will run for in secs. Defaults to 5.
        """
        if isinstance(goal_ee_pose, np.ndarray):
            goal_pose_fpy = RigidTransform(from_frame='franka_tool', 
                                    to_frame ='world',
                                    rotation = goal_ee_pose[0:3, 0:3],
                                    translation = goal_ee_pose[0:3, -1])   
        else:
            goal_pose_fpy = goal_ee_pose
        # logger.info(goal_pose_fpy)
        self.fpy_object.goto_pose(tool_pose = goal_pose_fpy, 
                                ignore_virtual_walls=ignore_virtual_walls, 
                                use_impedance=use_impedance, 
                                duration = duration)                 
        return

    def go_to_ee_pose_delta(self, delta_ee_pose, 
                                ignore_virtual_walls = True, 
                                use_impedance = False, 
                                duration = 3):
        """! Abstract function implementation of RobotArm Class. Moves the end-effector to a desired delta pose defined in the base frame

        @param    delta_ee_pose (numpy array): Desired 4x4 Transformation Matrix of desired delta pose from currrent End-Effector pose in the Robot base frame.
        @param    ignore_virtual_walls (bool, optional): Should frankapy ignore virtual walls. Defaults to True.
        @param    use_impedance (bool, optional): Should we run impedance control or position control. Defaults to False.
        @param    duration (int, optional): Frankapy needs to know the duration the joint position/impedance controller will run for in secs. Defaults to 3.
        """
        self.fpy_object.goto_pose_delta(delta_tool_pose = delta_ee_pose,
                                        ignore_virtual_walls = ignore_virtual_walls,
                                        use_impedance = use_impedance, 
                                        duration = duration
                                        )
        return

    def reset_arm(self,):
        """! resets arm to home joint configuration
        """
        self.fpy_object.reset_joints(self,)
    
    def open_gripper(self,):
        """! Open Franka gripper using frankapy
        """
        self.fpy_object.open_gripper(self,)
    
    def close_gripper(self,):
        """! Close Franka gripper using frankapy
        """
        self.fpy_object.close_gripper(self,)
    
    def get_randomized_absolute_poses(self, n_poses, flag_camera_in_hand = True):
        """! Pseudo radomly sample poses around the initial end-effector pose that gives good results for robot-camera calibration.
        This is achieved by very small translation and larger rotations round initial pose.

        @param    n_poses (int): number of end-effector poses to sample from
        @param    flag_camera_in_hand (bool, optional): Is a Camera attached to the robot hand or to its environment. Defaults to True.

        @return   RigidTransform list: list of randomly sample End-Effector poses around current end-effector pose
        """
        initial_pose = self.fpy_object.get_pose()
        from itertools import cycle
        toggle_signage = cycle([-1.0000,1.0000])
        def toggle_sign():
            return toggle_signage.__next__()

        ee_poses = []
        ee_poses.append(initial_pose)
        r, p ,y = ros_tf_utils.euler_from_matrix(initial_pose.rotation, 'rxyz')
        initial_translation = initial_pose.translation
        z_grid = np.array([1.0, 0.0, -1.0]) * 0.05
        signage = [-1.0000,1.0000]
        goal_pose = initial_pose.copy()
        for i in range(n_poses):
            x_offset = i % int(n_poses/len(z_grid))
            y_offset = i % int(n_poses/len(z_grid))
            sign = toggle_sign()
            x_offset *= 0.01 * random.choice(signage)
            y_offset *= 0.01 * random.choice(signage)    
            z_offset = random.choice(z_grid)            
            r_new = r + math.radians(random.uniform(0.0,30.0)) * random.choice(signage)
            p_new = p + math.radians(random.uniform(0.0,30.0)) * random.choice(signage)
            y_new = y + math.radians(random.uniform(0.0,80.0)) * random.choice(signage)
            goal_pose = RigidTransform(from_frame='franka_tool', 
                                    to_frame='world',
                                    rotation=ros_tf_utils.euler_matrix(r_new, p_new, y_new, 'rxyz')[0:3,0:3],
                                    translation=initial_pose.translation + np.array([x_offset, y_offset, z_offset]))
            ee_poses.append(goal_pose)
            # print(math.degrees(r_new),math.degrees(p_new),math.degrees(y_new),x_offset,y_offset,z_offset)

        return ee_poses

    # def get_randomized_absolute_poses(self, n_poses, flag_camera_in_hand = True):
    #     initial_pose = self.fpy_object.get_pose()
    #     from itertools import cycle
    #     toggle_signage = cycle([-1.0000,1.0000])
    #     def toggle_sign():
    #         return toggle_signage.__next__()

    #     ee_poses = []
    #     ee_poses.append(initial_pose)
    #     r, p ,y = ros_tf_utils.euler_from_matrix(initial_pose.rotation, 'rxyz')
    #     initial_translation = initial_pose.translation
    #     z_grid = np.array([1.0, 0.0, -1.0]) * 0.05
    #     signage = [-1.0000,1.0000]
    #     goal_pose = initial_pose.copy()
    #     for i in range(n_poses):
    #         x_offset = i % int(n_poses/len(z_grid))
    #         y_offset = i % int(n_poses/len(z_grid))
    #         sign = toggle_sign()
    #         x_offset *= 0.01 * random.choice(signage)
    #         y_offset *= 0.01 * random.choice(signage)    
    #         z_offset = random.choice(z_grid)            
    #         r_new = r + math.radians(random.uniform(25.0, 40.0)) * random.choice(signage)
    #         p_new = p + math.radians(random.uniform(25.0, 40.0)) * random.choice(signage)
    #         y_new = y + math.radians(random.uniform(50.0, 80.0)) * random.choice(signage)
    #         goal_pose = RigidTransform(from_frame='franka_tool', 
    #                                 to_frame='world',
    #                                 rotation=ros_tf_utils.euler_matrix(r_new, p_new, y_new, 'rxyz')[0:3,0:3],
    #                                 translation=initial_pose.translation + np.array([x_offset, y_offset, z_offset]))
    #         ee_poses.append(goal_pose)
    #     return ee_poses        

    
