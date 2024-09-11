#!/bin/bash
source ~/catkin_ws/devel_isolated/setup.bash 
roslaunch panda_moveit_config franka_control.launch robot_ip:=192.168.1.107 load_gripper:=true
