'''
author: Ruthrash Hari 
date: 02/10/23 
'''

'''
allows recording of desired grasps on objects by first performing pose estimation \
of the object and then recording the desired grasp obtained by moving the end effector manually. 
'''
import numpy as np
import cv2
import matplotlib.pyplot as plt
import dataclasses
from ..pick_and_place.object import transform_2_object_frame
from ..robot_camera_calibration._calibration_data_utils import tf_from_rvectvec, transform_between_cam_robot_frames, read_cam_robot_extrinsics
from ..config_reader import write_dict_to_yaml
from ..logger import logger

def draw_estimated_frame(color_image, camera, rvec, tvec):
    cv2.drawFrameAxes(color_image, camera.camera_matrix, camera.dist_coeffs, rvec, tvec, 0.05)
    return color_image

class RecordGrasps:
    def __init__(self, object, camera, extrinsics_file, robot_pose_collector, output_file, camera_in_hand = True) -> None:
        self.object = object    
        self.camera = camera
        self.extrinsics = read_cam_robot_extrinsics(extrinsics_file_name = extrinsics_file)
        # self.grasps_file = grasps_file
        self.camera_in_hand = camera_in_hand
        self.robot_pose_collector = robot_pose_collector
        self.output_file = output_file
        self.output_dict = dict()
        self.output_dict["object_name"] = self.object.object_name
        self.output_dict["grasps"] = dict()
        if self.object.tag is not None:
            self.output_dict["pose_estimator_data"] = dataclasses.asdict(self.object.tag.fiducial_data)
    
    def transform_2_object_frame(self, cam2base_tf, obj_rvec_in_cam, obj_tvec_in_cam, ee_rvec_in_base, ee_tvec_in_base):
        obj2cam_tf = tf_from_rvectvec(obj_rvec_in_cam, obj_tvec_in_cam)
        grasp2base_tf = tf_from_rvectvec(ee_rvec_in_base, ee_tvec_in_base)
        grasp2cam_tf = np.matmul(np.linalg.inv(cam2base_tf), grasp2base_tf)
        grasp2obj_tf = np.matmul(np.linalg.inv(obj2cam_tf), grasp2cam_tf)
        return grasp2obj_tf
    
    def get_cam2base_frame_tf(self,):
        if(self.camera_in_hand):
            ee2base_rvec, ee2base_tvec = self.robot_pose_collector.get_ee_frame()
            ee2base = tf_from_rvectvec(ee2base_rvec, ee2base_tvec)
            return np.matmul(ee2base, self.extrinsics)
        else: 
            return self.extrinsics 
    
    def plot_image_with_frame(self, color_image, frame2cam_rvec, frame2cam_tvec):
        color_image = color_image.copy()
        color_image = draw_estimated_frame(color_image = color_image,
                                                        camera = self.camera,
                                                        rvec = frame2cam_rvec,
                                                        tvec =  frame2cam_tvec)
        plt.imshow(color_image)
        plt.show(block=False)

    def run_grasp_recorder(self,):
        while(True):
            color_image = self.camera.get_current_rgb_frame()
            tag_rvec, tag_tvec, tag_detection = self.object.get_object_pose(color_image = self.camera.get_current_rgb_frame(), 
                                                                        camera = self.camera, 
                                                                        debug_image = True)
            # print tag detection variance
            self.plot_image_with_frame(color_image = color_image,
                                        frame2cam_rvec = tag_rvec,
                                        frame2cam_tvec = tag_tvec)     
            cam2base_view_pose = self.get_cam2base_frame_tf()        
            logger.info(f"Press Enter if you are happy with the current object's pose estimate. This is the rigid body frame with respect to which we will record grasps")
            ip = input()   
            if(ip != ""):
                continue
            logger.info(f"Now move the end effector to a desired grasp pose and press Enter when done")
            ip = input()         
            while(True):
                # query current grasp pose
                grasp2base_rvec, grasp2base_tvec = self.robot_pose_collector.get_ee_frame()
                # plot grasp pose in an image 
                grasp2cam_rvec, grasp2cam_tvec = transform_between_cam_robot_frames(cam2base_tf=cam2base_view_pose,
                                                                           rvec=grasp2base_rvec,
                                                                           tvec=grasp2base_tvec,
                                                                           to_camera_frame = True) 
                
                self.plot_image_with_frame(color_image = color_image,
                                           frame2cam_rvec = grasp2cam_rvec,
                                           frame2cam_tvec = grasp2cam_tvec)              
                logger.info(f"This is the grasp you have recorded, if you are happy with it press enter if not press space bar")
                ip = input()                
                if( ip!= ""):
                    continue    
                logger.info(f"Please enter a name for your grasp and press enter")
                grasp_name = input()                    
                self.output_dict["grasps"][grasp_name] = (transform_2_object_frame(cam2base_tf = cam2base_view_pose, 
                                                                                        obj_rvec_in_cam = tag_rvec, 
                                                                                        obj_tvec_in_cam = tag_tvec, 
                                                                                        ee_rvec_in_base = grasp2base_rvec, 
                                                                                        ee_tvec_in_base = grasp2base_tvec)).tolist()
                logger.info(f"Do you wish to record another grasp?, if yes, move the end effector to a desired grasp pose, press enter if not press space bar")             
                ip = input()                           
                if( ip!= ""):
                    break     
            break       
        n_grasps = len(self.output_dict["grasps"].keys())
        logger.info(f"Totally {n_grasps} new grasps have been recorded to file: {self.output_file}")
        write_dict_to_yaml(data_dict = self.output_dict, file_path = self.output_file)
        