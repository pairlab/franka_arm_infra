from pathlib import Path

# Third Party
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import cv2
import open3d as o3d
# torch.cuda.empty_cache()
# MegaPose
from megapose.config import LOCAL_DATA_DIR
from megapose.datasets.object_dataset import RigidObject, RigidObjectDataset
from megapose.datasets.scene_dataset import CameraData, ObjectData
from megapose.inference.types import (
    DetectionsType,
    ObservationTensor,
    PoseEstimatesType,
)
from megapose.inference.utils import make_detections_from_object_data
from megapose.lib3d.transform import Transform
from megapose.panda3d_renderer import Panda3dLightData
from megapose.panda3d_renderer.panda3d_scene_renderer import Panda3dSceneRenderer
from megapose.utils.conversion import convert_scene_observation_to_panda3d
from megapose.utils.load_model import NAMED_MODELS, load_named_model
from megapose.utils.logging import get_logger, set_logging_level
from megapose.visualization.bokeh_plotter import BokehPlotter
from megapose.visualization.utils import make_contour_overlay
from megapose.utils.tensor_collection import PandasTensorCollection

from ._object_pose_estimator import ObjectPoseEstimator
from ..camera.camera import get_bbox_annotations
from ..logger import logger
AVAILABLE_MODELS = ["megapose-1.0-RGB-multi-hypothesis", 
                    "megapose-1.0-RGB-multi-hypothesis-icp",
                    "megapose-1.0-RGBD",
                    "megapose-1.0-RGB"]

def rvectvec_from_tf(tf):
    """! Gets Angle-axis rotation vector and translation vector from 4x4 Transformation matrix.

    @param    tf (numpy array): 4x4 Transformation matrix.

    @param    numpy array: 3 dimensional angle-axis rotation vector. 
    @param    numpy array: 3 dimensional translation vector. 
    """
    rvec = cv2.Rodrigues(tf[0:3,0:3])[0]
    tvec = tf[0:3, -1]
    return rvec, tvec

def make_object_dataset(obj_dir_name: str) -> RigidObjectDataset:
    """! Creates and returns a RigidObjectDataset object that has the mesh for the physical object we want to do pose estimation for. 

    @param    obj_dir_name (str): Path to an object's meshes.

    @return    RigidObjectDataset: RigidObjectDataset class's object that contains mesh of the object who's pose we want to predict.
    """
    object_dir_path = Path(obj_dir_name)
    rigid_objects = []
    mesh_units = "m"
    object_dirs = (object_dir_path / "meshes").iterdir()
    for object_dir in object_dirs:
        label = object_dir.name
        mesh_path = None
        for fn in object_dir.glob("*"):
            if fn.suffix in {".obj", ".ply"}:
                assert not mesh_path, f"there multiple meshes in the {label} directory"
                mesh_path = fn
        assert mesh_path, f"couldnt find a obj or ply mesh for {label}"
        rigid_objects.append(RigidObject(label=label, mesh_path=mesh_path, mesh_units=mesh_units))
        # TODO: fix mesh units
    rigid_object_dataset = RigidObjectDataset(rigid_objects)
    return rigid_object_dataset

def load_observation_tensor(camera,
                            load_depth: bool = False)->ObservationTensor:
    """! Creates an observation tensor to be input to the neural network model from the RGB and Depth images from the camera. 

    @param    camera (Camera): Camera object associated with a real RGBDCamera to get current RGB or Depth image frames. 
    @param    load_depth (bool, optional): Is depth image also used?. Defaults to False.

    @return    bservationTensor: ObservationTensor object having RGB(D) image frames to run inference with MegaPose model. 
    """
    if load_depth: 
        rgb_image, depth_image = camera.get_current_rgbd_frames()
    else: 
        rgb_image = camera.get_current_rgb_frame() 
        depth_image = None
    return ObservationTensor.from_numpy(rgb_image, depth_image, 
                                        camera.camera_matrix).cuda(), rgb_image, depth_image

def load_detections(object_name, rgb_image, is_viz_crop = False) -> DetectionsType:
    """! Gets Bounding box of the physical object who's pose we want from the user. (can also modified to get from an object detection model)

    @param    object_name (str): Name of the physical object we are interested in. 
    @param    rgb_image (numpy array): 3 channgel RGB image as a matrix.
    @param    is_viz_crop (bool, optional): If true, the bounding box in the RGB image frame is rendered for visualization. Defaults to False.

    @return    DetectionsType: Bounding box information of the physical object we are interested in. 
    """
    bbox, cropped_image = get_bbox_annotations(rgb_image)
    infos = pd.DataFrame(
                dict(
                    label=[object_name],
                    batch_im_id=0,
                    instance_id=np.arange(1)
                )
            )
    bboxes = torch.as_tensor(
                np.stack([[bbox.xmin, bbox.ymin, 
                            bbox.xmax, bbox.ymax]]),
            )
    if is_viz_crop: 
        plt.imshow(cropped_image)
        plt.show(block=False)
        plt.pause(4.0)
        plt.close() 
        
    return PandasTensorCollection(infos=infos, bboxes=bboxes)

class MegaPose6D(ObjectPoseEstimator):#(object)
    """! ObjectPoseEstimator Abstract Class implementation for the MegaPose Object Pose Estimator https://github.com/megapose6d/megapose6d
    """
    def __init__(self,
                object_name,
                obj_dir_path:str,#.obj or .ply
                model_name = "megapose-1.0-RGB-multi-hypothesis-icp"):
        """! MegaPose6D class constructor. Loads pretrained MegaPose model. 

        
        @param    object_name (str): Name of the object whose pose we intend to estimate. 
        @param    obj_dir_path (str): Path to the directory that has the pretrained MegaPose model. 
        """
        ObjectPoseEstimator.__init__(self, "MegaPose6D")
        assert model_name in AVAILABLE_MODELS, "Specified model name is not available"
        self.model_info = NAMED_MODELS[model_name]
        self.obj_dir_path = obj_dir_path
        self.obj_dataset = make_object_dataset(self.obj_dir_path)
        self.object_name = object_name
        self.pose_estimator_model = load_named_model(model_name, self.obj_dataset).cuda()
        
    def __del__(self):
        """! MegaPose6D class destructor. Clears CUDA memory cache. 
        """
        logger.info("Calling megapose destructor")
        torch.cuda.empty_cache()
        
    def estimate_pose(self, camera, is_viz_prediction = True):
        """! Abstract method implemented for MegaPose6D. Uses Camera to get current RGB image frame and runs MegaPose6D inference model. 

        @param    camera (RGBCamera): RGBCamera Object that is used to get the current RGB image frame from camera.
        @param    is_viz_prediction (bool, optional): Should we visualize predited pose?. Defaults to True.

        @return    numpy array: 3x1 angle-axis rotation vector of the estimated pose.
        @return    numpy array: 3x1 translation vector of the estimated pose. 
        @return    _data_type_: final predictions object output by MegaPose6D
        """
        observation_tensor, rgb_image, depth_image = load_observation_tensor(camera,
                                                                load_depth=self.model_info["requires_depth"])
        
        detections = load_detections(self.object_name, rgb_image).cuda()
        
        output, _ = self.pose_estimator_model.run_inference_pipeline(
            observation_tensor, detections=detections, **self.model_info["inference_parameters"]
        )
        logger.info(output)
        
        rvec, tvec = rvectvec_from_tf(output.poses[0].cpu().detach().numpy())
        try:
            if is_viz_prediction:
                pcd = camera.get_pointcloud_rgbd(rgb_image, depth_image)
                object_mesh = o3d.io.read_triangle_mesh(self.obj_dir_path + "/meshes/" + self.object_name + "/textured.obj").transform(output.poses[0].cpu().detach().numpy())
                frame_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2).transform(output.poses[0].cpu().detach().numpy())
                
                # vis = o3d.visualization.Visualizer()
                # vis.create_window()
                # vis.add_geometry(pcd)
                # vis.add_geometry(frame_mesh)
                # vis.add_geometry(object_mesh)
                # vis.run()
                # sleep(3.0)
                # vis.destroy_window()
                o3d.visualization.draw_geometries([pcd, frame_mesh, object_mesh])
                
                # sleep(3.0)
                # vis.dery_window()
        except Exception as e: 
            logger.error(e)
            pass
        return rvec, tvec, output   