import yaml

from .camera.realsense_camera import RealSenseCamera
from .camera.ros_camera import ROSCamera
from .tags_detection.aruco_board import  ArucoBoardData
from .robot_camera_calibration.data_collector.arucoboard_data_collector import ArucoBoardDataCollector
from .robot_camera_calibration.data_collector.ros_robot_pose_collector import ROSRobotPoseCollector
from .robot_camera_calibration.data_collector.zmq_robot_pose_collector import ZmqRobotPoseCollector
from .logger import logger

def read_yaml_file(file_path):
    """! Reads a *.yaml or *.yml file.

    @param    file_path (str): Path of the YAML file.

    @return    dict: YAML data is returned as a dictionary.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def write_dict_to_yaml(data_dict, file_path):
    """Writes a dict object as a YAML file.

    @param    data_dict (dict): Dictionary to be written tot the file. 
    @param    file_path (str): Path of the YAML file.
    """
    with open(file_path, 'w') as file:
        yaml.dump(data_dict, file)

def get_camera(config_dict):
    """! Gets the appropriate Camera object based on the configuration file for robot camera calibration. 

    @param    config_dict (dict): robot camera calibraton configuration file that is loaded as a dict.

    @return    Camera: appropriate camera object based on configuration file.

    @exception    Exception: Failed to create setup configuration for calibration.
    @exception    KeyError : configuration yaml file for robot camera calibration is not properly setup.
    """
    if(config_dict["camera"] == "realsense_camera"):
        camera = RealSenseCamera()
    elif(config_dict["camera"] == "ros_camera"):
        try:       
            camera = ROSCamera(image_topic_name = config_dict["ros_camera"]["image_topic_name"],
                            camera_info_topic_name = config_dict["ros_camera"]["camera_info_topic"],
                            init_node = False)

        except Exception as error:
            logger.error(f"Failed to create setup configuration for calibration with error = {error}")
            return 
        except KeyError:
            logger.error(f"config yaml is not properly setup")
            return
    else:
        logger.error(f"you need to provide camera object in config file and also it should already be implemented")
        return
    return camera

def get_tag_pose_collector(config_dict):
    """! Gets the appropriate TagPoseCollector object based on the configuration file for robot camera calibration. 

    @param    config_dict (dict): robot camera calibraton configuration file that is loaded as a dict.

    @return    TagPoseCollector: appropriate TagPoseCollector object based on configuration file.

    @exception    Exception: Failed to create setup configuration for calibration.
    """
    if(config_dict["tag_pose_collector"] == "aruco_board"):    
        try:
            arucoboard_data = ArucoBoardData(dictionary = config_dict["aruco_board"]["dictionary"],
                                            marker_length = config_dict["aruco_board"]["marker_length"],
                                            marker_separation = config_dict["aruco_board"]["marker_separation"],
                                            n_rows = config_dict["aruco_board"]["n_rows"],      
                                            n_cols = config_dict["aruco_board"]["n_cols"])

            tag_pose_collector = ArucoBoardDataCollector(arucoboard_data, debug_detection_image = config_dict["debug_image"])
        except Exception as error:
            logger.error(f"Failed to create setup configuration for calibration with error = {error}")
            return 
    else: 
        logger.error(f"you need to provide tag_pose_collector object in config file and also it should already be implemented")
        return
    return tag_pose_collector

def get_robot_pose_collector(config_dict):
    """! Gets the appropriate RobotPoseCollector object based on the configuration file for robot camera calibration. 

    @param    config_dict (dict): robot camera calibraton configuration file that is loaded as a dict.

    @return    RobotPoseCollector: appropriate RobotPoseCollector object based on configuration file.

    @exception    Exception: Failed to create setup configuration for calibration
    """
    if(config_dict["robot_pose_collector"] == "ros_tf"):
        if(not config_dict["move_robot_automatically"]):
            logger.error(f"Cannot manually collect data if you want to get end-effector pose from a ROS TF tree")
            return
        logger.info("Ensure an active ROS TF of the robot arm is running")
        try:
            robot_pose_collector = ROSRobotPoseCollector(ee_frame = config_dict["ros_tf"]["ee_frame"],
                                                        base_frame = config_dict["ros_tf"]["base_frame"],
                                                        init_node = False)
        except Exception as error:
            logger.error(f"Failed to create setup configuration for calibration with error = {error}")
            return 
    elif(config_dict["robot_pose_collector"] == "zmq"):
        if(config_dict["move_robot_automatically"]):
            logger.error(f"Cannot automatically move robot collect data if you want to get end-effector pose using ZMQ server")
            return        
        logger.info("Ensure the zmq server is running in the realtime computer/docker")
        try:
            assert(config_dict["move_robot_automatically"] == False)
            robot_pose_collector = ZmqRobotPoseCollector(zmq_ip=config_dict["zmq"]["zmq_server_ip"],
                                                zmq_port=config_dict["zmq"]["zmq_server_port"])
        except Exception as error:
            logger.error(f"Failed to create setup configuration for calibration with error = {error}")
            return                                             
    else: 
        logger.error(f"you need to provide robot_pose_collector object in config file and also it should already be implemented")
        return

    return robot_pose_collector

def get_robot_camera_calib_config(config_dict):
    """! Gets all approprite objects required for robot camera calibration based on configuration described in YAML file located at "file_path"

    @param    config_dict (dict): Dictionary with the configuration information for robot camera calibration. 

    @return    TagPoseCollector: TagPoseCollector object that is used to get the calibration tag's pose in the camera frame.
    @return    RobotPoseCollector: RobotPoseCollector object that is used to get the robot's end-effector's pose in the robot's base frame. 
    @return    Camera: Camera object for the camera to be calibrated with robotarm.
    @return    dict: Dictionary containing the configuration of the robot camera calibration setup.
    """

    if(config_dict["only_calibration"]):
        return None, None, None
    
    camera = get_camera(config_dict)
    tag_pose_collector = get_tag_pose_collector(config_dict)
    robot_pose_collector = get_robot_pose_collector(config_dict)
    return tag_pose_collector, robot_pose_collector, camera


    