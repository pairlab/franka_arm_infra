import numpy as np
import cv2
import timeit
# Tensorflow

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
import matplotlib.pyplot as plt
import sys 
sys.path.insert(0, '/root/git/contact_graspnet/contact_graspnet')
 
from data import regularize_pc_point_count, depth2pc, load_available_input_data

from contact_grasp_estimator import GraspEstimator
from visualization_utils import visualize_grasps, show_image

from ..camera.camera import get_bbox_annotations, get_segmap_from_bbox_with_depth , get_segmap_from_bbox
from ._grasp_predictor import GraspPredictor
from ..logger import logger

class ContactGraspNet(GraspPredictor):
    """! GraspPredictor abstract class implementation for contact_graspnet https://github.com/NVlabs/contact_graspnet
    """
    def __init__(self,
                global_config,
                checkpoint_dir,
                z_range=[0.2,1.8],
                filter_grasps=True,
                local_regions=True,
                forward_passes=1,
                skip_border_objects=False): 
        """! ContactGraspNet Class Constructor. Loads the pretrained contact_graspnet model

        @param    global_config (dict): Configuration of the pretrained contact_graspnet model.
        @param    checkpoint_dir (str): Path to the pretrained contact_graspnet model.
        @param    z_range (list, optional): Range of depth measurements to be used for inference. Defaults to [0.2,1.8].
        @param    filter_grasps (bool, optional): If true, filter grasps to obtain contacts on specified point cloud segmentation. Defaults to True.
        @param    local_regions (bool, optional): If true, crop 3D local regions around object segments for prediction. Defaults to True.
        @param    forward_passes (int, optional): Number of forward passes to run on each point cloud. Defaults to 1.
        @param    skip_border_objects (bool, optional): If true, skip segments that are at the border of the depth map to avoid artificial edges. Defaults to False.
        """
        # Build the model
        self.grasp_estimator = GraspEstimator(global_config)
        self.grasp_estimator.build_network()
        self.z_range = z_range
        self.skip_border_objects = skip_border_objects
        self.forward_passes = forward_passes
        self.filter_grasps = filter_grasps
        # Add ops to save and restore all the variables.
        saver = tf.train.Saver(save_relative_paths=True)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        self.sess = tf.Session(config=config)
        self.local_regions = local_regions
        # Load weights
        self.grasp_estimator.load_weights(self.sess, saver, checkpoint_dir, mode='test')

    def generate_grasps(self, rgbd_camera, is_visualize_grasps = False, use_depth_for_seg = False):
        """! Abstract method implemented for contact_graspnet. Gets Segmentation mask for the part of the depth image to run grasp prediction

        @param    rgbd_camera (RGBDCamera): RGBDCamera Class object to get current RGB and Depth image frames. 
        @param    is_visualize_grasps (bool, optional): If true, visualizes the predicted grasps overlayed on the pointcloud of the scene. Defaults to False.
        @param    use_depth_for_seg (bool, optional): If true, uses average depth of the scene to filter out the background of the depth image and runs inference on the foreground. Defaults to False.

        @return   numpy array: predict grasp poses
        @return   numpy array: scores for predicted grasps
        @return   numpy array: contact points
        """
        rgb_image, depth_image = rgbd_camera.get_current_rgbd_frames()
        bbox, _ = get_bbox_annotations(rgb_image)
        if use_depth_for_seg:
            segmap = get_segmap_from_bbox_with_depth(rgb_image, depth_image, bbox)
        else:            
            segmap = get_segmap_from_bbox(rgb_image, bbox)        
        # 
        plt.imshow(segmap)
        plt.show()    
        cam_k = rgbd_camera.camera_matrix
        start = timeit.timeit()
        pc_full, pc_segments, pc_colors = self.grasp_estimator.extract_point_clouds(depth_image, 
                                                                                    cam_k, 
                                                                                    segmap=segmap,
                                                                                    rgb=rgb_image,
                                                                                    skip_border_objects=self.skip_border_objects,
                                                                                    z_range=self.z_range)
        pred_grasps_cam, scores, contact_pts, _ = self.grasp_estimator.predict_scene_grasps(self.sess, pc_full, 
                                                                                            pc_segments=pc_segments,
                                                                                            local_regions=self.local_regions, 
                                                                                            filter_grasps=self.filter_grasps, 
                                                                                            forward_passes=self.forward_passes)  
        end = timeit.timeit()
        logger.info(f"elapsed time {end-start}")
 
        show_image(rgb_image, segmap)
        plt.imshow(segmap)
        plt.show()
        if is_visualize_grasps:
            visualize_grasps(pc_full, pred_grasps_cam, scores, plot_opencv_cam=True, pc_colors=pc_colors)
        return pred_grasps_cam, scores, contact_pts#, _

    
    