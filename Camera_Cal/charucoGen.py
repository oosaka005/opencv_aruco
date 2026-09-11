# This script generates a ChArUco board image and saves it as 'charuco.png'.

import cv2
import numpy as np
import sys
import time
from PIL import Image

# Define parameters
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
# SQUARES_VERTICALLY = 6
# SQUARES_HORIZONTALLY = 8
# SQUARE_LENGTH = 0.021166  # meters
# MARKER_LENGTH = 0.016 # meters
# SQUARES_VERTICALLY = 4
# SQUARES_HORIZONTALLY = 6
# SQUARE_LENGTH = 0.0254  # meters
# MARKER_LENGTH = 0.020 # meters
SQUARES_VERTICALLY = 3
SQUARES_HORIZONTALLY = 4
SQUARE_LENGTH = 0.0381  # meters
MARKER_LENGTH = 0.029575 # meters

board = cv2.aruco.CharucoBoard((SQUARES_HORIZONTALLY, SQUARES_VERTICALLY), 
                               SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT)   

# img = board.generateImage((2000, 1500))
img = board.generateImage((1800, 1350))

# Convert BGR (OpenCV) to RGB (PIL)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Save with PIL, specifying DPI as 300
pil_img = Image.fromarray(img_rgb)
pil_img.save('charuco.png', dpi=(300, 300))

url = "http://192.168.10.141:5000"


