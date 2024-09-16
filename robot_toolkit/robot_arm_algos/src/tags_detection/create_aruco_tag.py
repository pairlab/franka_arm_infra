import cv2 
from ._fiducial import ARUCO_DICT
from ...src.logger import logger

def create_and_save_arucotag(output_image_file, dictionary, id_ = 0):
    """! Creates an image with the arucotag of appropriate dictionary and id

    @param    output_image_file (str): path to the file in which the image is to be saved (*.jpg or *.png)
    @param    dictionary (str): ArucoDictionary of the marker  
    @param    id_ (int, optional): ID of the marker to be rendered. Defaults to 0.

    @return    bool: If saving the image with arucotag pattern was successful or not
 
    @exception    Exception: KeyError if the requested dictionary doesn't exist in available dictionaries of aruco markers.KeyError
    """
    try:
        cv_dictionary = ARUCO_DICT[dictionary]
    except KeyError:
        logger.error(f"{dictionary} that was provided doesn't exist in the list of options: {list(ARUCO_DICT.keys())}" )
        return
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv_dictionary)
    color_image = cv2.aruco.generateImageMarker(aruco_dict, id_, sidePixels=3300)
    return cv2.imwrite(output_image_file, color_image)

def create_and_save_arucoboard(output_image_file, dictionary, n_rows, n_cols, marker_length, marker_separation, ids = None):
    """! Creates an image with the arucoboard of appropriate dictionary, number of rows and 
    columns, marker length and the separation between markers in the arucoboard

    @param    output_image_file (str): path to the file in which the image is to be saved (*.jpg or *.png)
    @param    dictionary (str): ArucoDictionary of the markers in the aruco board
    @param    n_rows (int): number of rows of aruco markers in the board 
    @param    n_cols (int): number of columns of aruco markers in the board
    @param    marker_length (float): edge length of the square aruco marker in meters
    @param    marker_separation (float): distance between two square aruco markers in the board
    @param    ids(list, Optional): You can specify the ID of the markers to be rendered in the ArucoBoard.Defaults to None

    @return    bool: If saving the image with arucotag pattern was successful or not

    @exception    Exception: KeyError if the requested dictionary doesn't exist in available dictionaries of aruco markers.KeyError
    """
    try:
        cv_dictionary = ARUCO_DICT[dictionary]
    except KeyError:
        logger.error(f"{dictionary} that was provided doesn't exist in the list of options: {list(ARUCO_DICT.keys())}" )
        return
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv_dictionary)
    board = cv2.aruco.GridBoard((n_rows, n_cols), 
                                marker_length, 
                                marker_separation, 
                                aruco_dict,
                                ids = ids)    
    color_image = None
    color_image = board.generateImage((3300,3300), color_image)
    return cv2.imwrite(output_image_file, color_image)  