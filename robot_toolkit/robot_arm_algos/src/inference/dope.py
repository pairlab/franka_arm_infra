import yaml
import numpy as np
import cv2
import tf.transformations
# from ..config_reader import  read_yaml_file
from dope.inference.cuboid import Cuboid3d
from dope.inference.cuboid_pnp_solver import CuboidPNPSolver
from dope.inference.detector import ModelData, ObjectDetector

from PIL import ImageDraw
from PIL import Image

from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import PoseStamped
from ._object_pose_estimator import ObjectPoseEstimator
from ..logger import logger

def rotate_vector(vector, quaternion):
    """! Using a given quaternion rotation, rotate the given vector

    @param    vector (numpy array): 3 dimension translation vector
    @param    quaternion (numpy array): 4 dimennsional quaternion rotation vector

    @return    numpy array: 3 dimensional rotated vector
    """
    q_conj = tf.transformations.quaternion_conjugate(quaternion)
    vector = np.array(vector, dtype='float64')
    vector = np.append(vector, [0.0])
    vector = tf.transformations.quaternion_multiply(q_conj, vector)
    vector = tf.transformations.quaternion_multiply(vector, quaternion)
    return vector[:3]

def read_yaml_file(file_path):
    """! Reads a YAML file and returns its content as a dictionary. 

    @param    file_path (str): Path to the YAML file. 

    @return  dict: Content in the YAML file as a dictionary.
    """
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def draw_estimated_frame(color_image, camera, rvec, tvec):
    """! Draws a frame of the DOPE's predicted object pose in the camera frame.

    @param    color_image (numpy array): 3 channel RGB image on which the detections will be drawn
    @param    camera (Camera): Camera Object that has the intrinsics and distortion parameters.
    @param    rvec (numpy array): 3x1 angle-axis rotation vector of the object's pose in the camera frame.
    @param    tvec (numpy array): 3x1 translation vector of the object's pose in the camera frame.

    @return    numpy array: 3 channel RGB image which has arucboard's estimated frame drawn on it.
    """    
    cv2.drawFrameAxes(color_image, camera.camera_matrix, camera.dist_coeffs, rvec, tvec, 0.1)
    return color_image
    
class DOPEPoseDetection(ObjectPoseEstimator):#(object)
    """! ObjectPoseEstimator Abstract Class implementation for the DOPE Object Pose Estimator https://github.com/NVlabs/Deep_Object_Pose
    """
    def __init__(self,
                model_weights_path,
                dope_config_pose_file,
                object_model_name = "soup",
                input_is_rectified = True,
                overlay_belief_images = True,
                downscale_height = 500,
                mask_edges = 1,
                mask_faces = 1, 
                vertex = 1,
                threshold = 0.5,
                softmax = 1000,
                thresh_angle = 0.5, 
                thresh_map = 0.01, 
                sigma = 3, 
                thresh_points = 0.1):
        """! DOPEPoseDetection Class Constructor gets the pose detector model's configuration and loads the pretrained DOPE model. 

        @param    model_weights_path (str): Path to the DOPE pretrained model.
        @param    dope_config_pose_file (str): Path to the pretrained DOPE model's configuration file. 
        @param    object_model_name (str, optional): DOPE models are pretrained for each YCB object, this strip specifies which object. Defaults to "soup".
        @param    input_is_rectified (bool, optional): Is the input RGB image rectified from distortion?. Defaults to True.
        @param    overlay_belief_images (bool, optional): Should we overlay belief images to the input image?. Defaults to True.
        @param    downscale_height (int, optional): If the input image is larger than this, scale it down to this pixel height. Very large input images eat up all the GPU memory and slow down inference. Also, DOPE works best when the object size (in pixels) has appeared in the training data (which is downscaled to 400 px). For these reasons, downscaling large input images to something reasonable (e.g., 400-500 px) improves memory consumption, inference speed *and* recognition results. Defaults to 500.
        @param    mask_edges (int, optional): Undocumented in DOPE. Defaults to 1.
        @param    mask_faces (int, optional): Undocumented in DOPE. Defaults to 1.
        @param    vertex (int, optional): Undocumented in DOPE. Defaults to 1.
        @param    threshold (float, optional): Undocumented in DOPE. Defaults to 0.5.
        @param    softmax (int, optional): Undocumented in DOPE. Defaults to 1000.
        @param    thresh_angle (float, optional): Undocumented in DOPE. Defaults to 0.5.
        @param    thresh_map (float, optional): Undocumented in DOPE. Defaults to 0.01.
        @param    sigma (int, optional): Undocumented in DOPE. Defaults to 3.
        @param    thresh_points (float, optional): Undocumented in DOPE. Defaults to 0.1.
        """
        ObjectPoseEstimator.__init__(self, "DOPE")
        
        self.input_is_rectified = input_is_rectified
        self.downscale_height = downscale_height
        self.overlay_belief_images = overlay_belief_images
        
        self.config_detect = lambda: None        
        self.config_detect.mask_edges = mask_edges
        self.config_detect.mask_faces = mask_faces
        self.config_detect.vertex = vertex 
        self.config_detect.threshold = threshold 
        self.config_detect.softmax = softmax 
        self.config_detect.thresh_angle = thresh_angle
        self.config_detect.thresh_map = thresh_map 
        self.config_detect.sigma = sigma 
        self.config_detect.thresh_points = thresh_points 
        
        self.config_dope_dict = read_yaml_file(dope_config_pose_file)
        model = object_model_name
        weights_url = self.config_dope_dict["weights"][model]
        
        self.dnn_model = ModelData(model,
                                    model_weights_path)
        self.dnn_model.load_net_model()        
        try: 
            M = np.array(self.config_dope_dict["model_transform"][model], dtype='float64')
            self.model_transform = tf.transformations.quaternion_from_matrix(M)      
        except KeyError:
            self.model_transform = np.array([0.0, 0.0, 0.0, 1.0], dtype='float64')

        try: 
            self.mesh = self.config_dope_dict["meshes"][model]
        except KeyError:
            pass
        
        try: 
            self.mesh_scale = self.config_dope_dict["mesh_scales"][model]
        except KeyError:
            self.mesh_scale = 1.0  

        try:
            self.draw_colors = tuple(self.config_dope_dict["draw_colors"][model])
        except:
            self.draw_colors = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        
        self.dimension = tuple(self.config_dope_dict["dimensions"][model])         
        self.class_id = self.config_dope_dict["class_ids"][model]
        self.pnp_solver = CuboidPNPSolver(model,
                                        cuboid3d=Cuboid3d(self.config_dope_dict["dimensions"][model]))

    def estimate_pose(self, camera, save_image_file_path=None):
        """! Abstract method implemented for DOPE. Uses Camera to get current RGB image frame and runs DOPE inference model. 

        @param    camera (RGBCamera): RGBCamera Object that is used to get the current RGB image frame from camera
        @param    save_image_file_path (str, optional): file name including path to store image overlayed with predicted object pose. Defaults to None.
        """
        rgb_image = camera.get_current_rgb_frame()
        camera_matrix = camera.camera_matrix 
        dist_coeffs = camera.dist_coeffs
        height, width, _ = rgb_image.shape
        scaling_factor = float(self.downscale_height) / height
        if scaling_factor < 1.0: 
            camera_matrix[:2] *= scaling_factor
            rgb_image = cv2.resize(rgb_image, (int(scaling_factor * width), int(scaling_factor * height)))
        self.pnp_solver.set_camera_intrinsic_matrix(camera_matrix)
        self.pnp_solver.set_dist_coeffs(dist_coeffs) 
        
        rgb_image_copy = rgb_image.copy() 
        im = Image.fromarray(rgb_image_copy)
        draw = Draw(im)
        results, im_belief = ObjectDetector.detect_object_in_image(
                self.dnn_model.net,
                self.pnp_solver,
                rgb_image,
                self.config_detect,
                make_belief_debug_img=True,
                overlay_image=self.overlay_belief_images
            )
        # Publish pose and overlay cube on image
        for i_r, result in enumerate(results):
            if result["location"] is None:
                continue
            loc = result["location"]
            ori = result["quaternion"]

            # transform orientation
            transformed_ori = tf.transformations.quaternion_multiply(ori, self.model_transform)

            dims = rotate_vector(vector=self.dimension, quaternion=self.model_transform)
            dims = np.absolute(dims)
            dims = tuple(dims)
            pose_msg = PoseStamped()
            # pose_msg.header = image_msg.header
            CONVERT_SCALE_CM_TO_METERS = 100
            pose_msg.pose.position.x = loc[0] / CONVERT_SCALE_CM_TO_METERS
            pose_msg.pose.position.y = loc[1] / CONVERT_SCALE_CM_TO_METERS
            pose_msg.pose.position.z = loc[2] / CONVERT_SCALE_CM_TO_METERS
            pose_msg.pose.orientation.x = transformed_ori[0]
            pose_msg.pose.orientation.y = transformed_ori[1]
            pose_msg.pose.orientation.z = transformed_ori[2]
            pose_msg.pose.orientation.w = transformed_ori[3]

            # Draw the cube
            if None not in result['projected_points'] and save_image_file_path is not None:
                points2d = []
                for pair in result['projected_points']:
                    points2d.append(tuple(pair))
                draw.draw_cube(points2d, self.draw_colors)            
                logger.info("Image saved at " + save_image_file_path)
                im.save(save_image_file_path)
            
class Draw(object):
    """! Drawing helper class to visualize the DOPE neural network output"""
    def __init__(self, im):
        """! Draw Class Constructor
        @param im(numpy array): The image to draw in.
        """
        self.draw = ImageDraw.Draw(im)

    def draw_line(self, point1, point2, line_color, line_width=2):
        """! Draws line on image using PIL module 
        
        @param    point1(numpy array): image coordinate of line's start.
        @param    point2(numpy array): image coordinate of line's end.
        @param    line_color(numpy array): 3 dimensional array of RGB color values.
        @param    line_width (int, optional): Width of the drawn line. Defaults to 2.
        """
        if point1 is not None and point2 is not None:
            self.draw.line([point1, point2], fill=line_color, width=line_width)

    def draw_dot(self, point, point_color, point_radius):
        """! Draws dot (filled circle) on image

        @param    point (numpy array): image coordinate of point.
        @param    point_color (numpy array): 3 dimensional array of RGB color values.
        @param    point_radius (float): radius of point to be drawnn.
        """

        if point is not None:
            xy = [
                point[0] - point_radius,
                point[1] - point_radius,
                point[0] + point_radius,
                point[1] + point_radius
            ]
            self.draw.ellipse(xy,
                              fill=point_color,
                              outline=point_color
                              )

    def draw_cube(self, points, color=(255, 0, 0)):
        """! Draws cube with a thick solid line across
        the front top edge and an X on the top face.

        @param    points (list): list of points corresponding to vertices of the cube
        @param    color (tuple, optional): 3 dimensional array of RGB color values. Defaults to (255, 0, 0).
        """

        # draw front
        self.draw_line(points[0], points[1], color)
        self.draw_line(points[1], points[2], color)
        self.draw_line(points[3], points[2], color)
        self.draw_line(points[3], points[0], color)

        # draw back
        self.draw_line(points[4], points[5], color)
        self.draw_line(points[6], points[5], color)
        self.draw_line(points[6], points[7], color)
        self.draw_line(points[4], points[7], color)

        # draw sides
        self.draw_line(points[0], points[4], color)
        self.draw_line(points[7], points[3], color)
        self.draw_line(points[5], points[1], color)
        self.draw_line(points[2], points[6], color)

        # draw dots
        self.draw_dot(points[0], point_color=color, point_radius=4)
        self.draw_dot(points[1], point_color=color, point_radius=4)

        # draw x on the top
        self.draw_line(points[0], points[5], color)
        self.draw_line(points[1], points[4], color)