from abc import ABC, abstractmethod

class GraspPredictor(ABC):
    """! Abstract class for running inference on grasp predictor models that take RGB or Depth or RGBD images input
    """
    def __init__(self, algorithm_name=None):
        """! Abstract GraspPredictor class' constructor

        @param    algorithm_name (str, optional): Name of the method/algoritm for grasp prediction. Defaults to None.
        """
        self.algorithm_name = algorithm_name

    @abstractmethod
    def generate_grasps(self, camera):
        """! Abstract method that needs to be implemented for any grasp prediction algorithm

        @param    camera (Camera): RGB or RGBD or Depth camera object who's sensor data is used for inference
        """
        pass