from dataclasses import dataclass
import numpy as np
import pyrealsense2 as rs
from ..logger import logger

"""!
Data class to store Camera Intrinsic information
"""
@dataclass
class Intrinsics:
    """! A data class for the Camera Intrinsic Parameters 
    """    
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

class RealSenseDriver(object):
    """! A driver Class for Intel RealSense D415/D435 cameras.
    """

    def __init__(self, serial_number, 
                laser_power=16, 
                accuracy=1,
                color_width = 1280,
                color_height = 720,
                color_fps = 30,
                depth_width = 1280,
                depth_height = 720,
                depth_fps = 30,
                motion_range=0, 
                filter_option=3, 
                confidence_threshold=1,
                near_threshold=0.1, 
                far_threshold=6.0, 
                use_filters=True,
                only_depth=False):
        """! RealSenseDriver Class Constructors. Allows setting custom Camera settings

        
        @param    serial_number (int): realsense camera's serial number
        @param    laser_power (int, optional): level of IR pattern laser's power. Defaults to 16.
        @param    accuracy (int, optional): accuracy option of realsense cameras. Defaults to 1.
        @param    color_width (int, optional): width of the color image. Defaults to 1280.
        @param    color_height (int, optional): height of the color image. Defaults to 720.
        @param    color_fps (int, optional): FPS of the color camera. Defaults to 30.
        @param    depth_width (int, optional): width of the depth image. Defaults to 1280.
        @param    depth_height (int, optional): height of the depth image. Defaults to 720.
        @param    depth_fps (int, optional): FPS of the depth camera. Defaults to 30.
        @param    motion_range (int, optional): motion_range option of realsense cameras. Defaults to 0.
        @param    filter_option (int, optional): filter_option of realsense cameras. Defaults to 3.
        @param    confidence_threshold (int, optional): confidence_threshold option for realsense cameras. Defaults to 1.
        @param    near_threshold (float, optional): near_threshold option for realsense cameras. Defaults to 0.1.
        @param    far_threshold (float, optional): far_threshold option for realsense cameras. Defaults to 6.0.
        @param    use_filters (bool, optional): use_filters option for realsense cameras. Defaults to True.
        @param    only_depth (bool, optional): only_depth option for realsense cameras, if true only depth camera is powered ON. Defaults to False.
        """
        self.serial_number = serial_number
        self.laser_power = laser_power
        self.accuracy = accuracy#
        self.motion_range = motion_range#
        self.filter_option = filter_option#
        self.confidence_threshold = confidence_threshold#
        self.near_threshold = near_threshold
        self.far_threshold = far_threshold
        self.use_filters = use_filters
        self.only_depth = only_depth

        self._pipeline = None
        self._profile = None
        self._scale = None
        self._is_running = False
        self._filters = []
        self.color_width = color_width
        self.color_height = color_height
        self.color_fps = color_fps
        self.depth_width = depth_width
        self.depth_height = depth_height
        self.depth_fps = depth_fps

    @property
    def serial_number(self):
        """! Gets Camera's serial number


        @return str: camera serial number
        """
        return self._serial_number

    @serial_number.setter
    def serial_number(self, value):
        self._serial_number = str(value)

    @property
    def laser_power(self):
        """! Gets Cameras laser power. Only works for D415/D435(i)


        @return int: Power of laser to use in mW.
        """
        return self._laser_power

    @laser_power.setter
    def laser_power(self, value):
        self._laser_power = int(value)

    @property
    def accuracy(self):
        """! int: .
        """
        return self._accuracy

    @accuracy.setter
    def accuracy(self, value):
        self._accuracy = value

    @property
    def motion_range(self):
        """! int: .
        """
        return self._motion_range

    @motion_range.setter
    def motion_range(self, value):
        self._motion_range = value

    @property
    def filter_option(self):
        """! int: .
        """
        return self._filter_option

    @filter_option.setter
    def filter_option(self, value):
        self._filter_option = value

    @property
    def confidence_threshold(self):
        """! int: .
        """
        return self._confidence_threshold

    @confidence_threshold.setter
    def confidence_threshold(self, value):
        self._confidence_threshold = value

    @property
    def near_threshold(self):
        """! int: .
        """
        return self._near_threshold

    @near_threshold.setter
    def near_threshold(self, value):
        self._near_threshold = value

    @property
    def far_threshold(self):
        """! int: .
        """
        return self._far_threshold

    @far_threshold.setter
    def far_threshold(self, value):
        self._far_threshold = value

    @property
    def use_filters(self):
        """! bool: Whether to use simple filters to improve depth quality.
        """
        return self._use_filters

    @use_filters.setter
    def use_filters(self, value):
        self._use_filters = value

    def start(self):
        """! Start the sensor driver.
        

        @exception Exception: If unable to detect/power ON the realsense camera
        """
        if self._is_running:
            return True

        logger.info('Starting Intel RealSense {}'.format(self.serial_number))
        try:
            self._pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.serial_number)
            config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)            
            # config.enable_stream(rs.stream.depth, self.color_height, self.color_width, rs.format.z16, self.color_fps)
            # config.enable_stream(rs.stream.color, self.depth_height, self.depth_width, rs.format.rgb8, self.depth_fps)
            self._profile = self._pipeline.start(config)

            # Set settings
            device = self._profile.get_device()
            ds = device.first_depth_sensor()
            ds.set_option(rs.option.laser_power, self.laser_power)
            # print(ds.get_option(rs.option.laser_power))
            # ds.set_option(rs.option.accuracy, rs.option.accuracy)
            # print(ds.get_option(rs.option.accuracy))
            # ds.set_option(rs.option.motion_range, self.motion_range)
            # ds.set_option(rs.option.filter_option, self.filter_option)
            # ds.set_option(rs.option.confidence_threshold, self.confidence_threshold)
            self._scale = ds.get_depth_scale()
            sensors = device.query_sensors()
            # for s in sensors:
                # if not s.is_depth_sensor():
                # if s.is_color_sensor():                
                    # print(s)
                    # s.set_option(rs.option.exposure, 3500)
            
            
        except Exception as e:
            logger.error(e)
            raise Exception('Failed to start Intel RealSense {}'.format(
                self.serial_number
            ))

        if self.use_filters:
            self._filters = [#rs.decimation_filter(),
                             rs.threshold_filter(min_dist=self.near_threshold,
                                                 max_dist=self.far_threshold),]
                            #  rs.spatial_filter(),
                            #  rs.temporal_filter()]

        # Read the first frames to clear any bad frames
        for _ in range(20):
            _ = self._pipeline.wait_for_frames()
        
        self._is_running = True
        return self._is_running

    def stop(self):
        """! Stop the sensor driver.
        """
        if not self._is_running:
            logger.warning('Intel RealSense not running, cannot stop')
            return False
        self._pipeline.stop()
        self._pipeline = None
        self._profile = None
        self._scale = None
        self._filters = []
        self._is_running = False
        return self._is_running

    def read(self):
        """! Read data from the sensor and return it.

        
        @exception    Exception: Failed to read frames from Intel RealSense Camera
        @exception    Exception: RealSense Camera not connected
            
        @return    uint8 numpy array (3,h,w):  The color data.
        @return    float numpy array (h,w): The depth data
        """

        if not self._is_running:
            raise Exception('Intel RealSense {} not connected'.format(
                self.serial_number
            ))

        try:
            frames = self._pipeline.wait_for_frames()
            if not self.only_depth:
                align = rs.align(rs.stream.color)
                frames = align.process(frames)
            depth = frames.get_depth_frame()
            if self.use_filters:
                for f in self._filters:
                    depth = f.process(depth)
            depth = self._scale * np.array(depth.as_frame().get_data())

            if not self.only_depth:
                color = frames.get_color_frame()
                color = np.array(color.as_frame().get_data())
                depth = depth.astype(np.float32)                
                return color, depth
            else:
                depth = depth.astype(np.float32)                
                return depth
        except Exception as e:
            logger.error(e)
            raise Exception('Failed to read frames from Intel RealSense '
                            '{}'.format(self.serial_number))

    def get_intrinsics(self):
        """! Get the intrinsics from the running sensor.


        @return  Intrinsics: The intrinsics for the active sensor.
        

        @exception    Exception: Failed to read frames from Intel RealSense Camera
        """
        if not self._is_running:
            raise Exception('Intel RealSense {} not connected'.format(
                self.serial_number
            ))
        try:
            frames = self._pipeline.wait_for_frames()
            if not self.only_depth:
                align = rs.align(rs.stream.color)
                frames = align.process(frames)
            depth = frames.get_depth_frame()
            i = rs.video_stream_profile(depth.profile).get_intrinsics()
            return Intrinsics(fx=i.fx, fy=i.fy, cx=i.ppx, cy=i.ppy,
                              width=i.width, height=i.height)
        except Exception as e:
            logger.error(e)
            raise Exception('Failed to read frames from Intel RealSense '
                            '{}'.format(self.serial_number))

    @staticmethod
    def get_all_device_serials():
        """! Get the serial numbers for all connected sensors.

        return    str list: The serial numbers.
        """
        ctx = rs.context()
        serials = []
        for d in ctx.devices:
            serials.append(str(d.get_info(rs.camera_info.serial_number)))
        return serials