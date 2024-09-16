from abc import ABC, abstractmethod

class ObjectPoseEstimator(ABC):
    """! Abstract Class for running inference on object pose estimation models
    """
    def __init__(self, algorithm_name=None):
        """! Abstract ObjectPoseEstimator Class Constructor
        
        @param algorithm_name (str, optional): Name of the object pose estimation algorithm. Defaults to None.
        """
        self.algorithm_name = algorithm_name

    @abstractmethod
    def estimate_pose(self, camera):
        """! Abstract Method that needs to be implemented for object pose estimation methods
        
        @param camera (Camera): RGB or RGBD or Depth camera object who's sensor data is used for inference
        """
        pass