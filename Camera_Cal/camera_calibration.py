# Camera calibration using ArUco markers and Charuco board
# This script processes images captured for calibration, detects ArUco markers, and extracts Charuco corners for camera calibration.
# Run on Laptop with OpenCV installed, and ensure calibration images are in the 'calibration_images' folder. Results will be saved in 'calibration_results'.

import cv2
import numpy as np
import sys
import time
import os

#File Paths
image_folder = os.path.join(os.path.dirname(__file__), 'cal_images')
os.makedirs(image_folder, exist_ok=True)  # Create folder if it doesn't exist
output_folder = os.path.join(os.path.dirname(__file__), 'output_images')
os.makedirs(output_folder, exist_ok=True)  # Create folder if it doesn't exist

# Define parameters
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
ARUCO_PARAMS = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(ARUCO_DICT, ARUCO_PARAMS)

# Optional: Adjust parameters for better detection
# ARUCO_PARAMS.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
# ARUCO_PARAMS.adaptiveThreshWinSizeMin = 3
# ARUCO_PARAMS.adaptiveThreshWinSizeMax = 23
# ARUCO_PARAMS.adaptiveThreshWinSizeStep = 10
# ARUCO_PARAMS.adaptiveThreshConstant = 0.03

# For charuco board definition
SQUARES_VERTICALLY = 6
SQUARES_HORIZONTALLY = 8
SQUARE_LENGTH = 0.02116  # meters
MARKER_LENGTH = 0.016 # meters

board = cv2.aruco.CharucoBoard((SQUARES_HORIZONTALLY, SQUARES_VERTICALLY), 
                               SQUARE_LENGTH, MARKER_LENGTH, ARUCO_DICT)

# Create CharUco detector (new API for OpenCV 4.7.0+)
charuco_params = cv2.aruco.CharucoParameters()
charuco_detector = cv2.aruco.CharucoDetector(board, charuco_params, ARUCO_PARAMS)

all_charuco_corners = []
all_charuco_ids = []

# Process each image in the calibration_images folder for detection of markers
for image in os.listdir(image_folder):
    if image.endswith('.jpg'):
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error loading image: {image_path}")
            continue
        
        # Detect ChArUco corners using the new API (OpenCV 4.7.0+)
        charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(frame)

        # Handle None values from detection
        if marker_ids is None:
            marker_ids = []
        if charuco_ids is None:
            charuco_ids = []
        
        print(f"Processing {image}: Detected {len(marker_ids)} ArUco markers, {len(charuco_ids)} ChArUco corners")
        
        # Handle None values for charuco_corners
        if charuco_corners is None:
            charuco_corners = []
        
        if len(charuco_ids) > 0 and len(charuco_corners) > 20:  # Ensure we have enough corners for calibration
            print(f"ChArUco corners detected: {len(charuco_corners)}")
            # Draw detected markers on the image
            if marker_ids is not None:
                frame_marked = cv2.aruco.drawDetectedMarkers(frame.copy(), marker_corners, marker_ids)
            else:
                frame_marked = frame.copy()
            
            # Draw charuco corners for visualization
            frame_marked = cv2.aruco.drawDetectedCornersCharuco(frame_marked, charuco_corners, charuco_ids)
            cv2.imwrite(os.path.join(output_folder, f"charuco_{image}"), frame_marked)  # Save charuco marked image

            # Append corners and IDs for calibration
            all_charuco_corners.append(charuco_corners)
            all_charuco_ids.append(charuco_ids)
        else:
            print(f"  Not enough ChArUco corners detected for {image}")

# Perform camera calibration once after collecting all corners
print("\n" + "="*60)
print("Performing camera calibration...")
print("="*60)

if len(all_charuco_corners) > 0:
    # Get image size from first image
    first_image_path = os.path.join(image_folder, [f for f in os.listdir(image_folder) if f.endswith('.jpg')][0])
    first_frame = cv2.imread(first_image_path)
    imageSize = first_frame.shape[1::-1]  # (width, height)
    
    # Convert ChArUco corners to object points using the board
    objPoints = []
    imgPoints = []
    
    for i in range(len(all_charuco_corners)):
        # # Manual method (equivalent to matchImagePoints below)
        # corners_3d = board.getChessboardCorners()
        # matched_obj_points = []
        # matched_img_points = []
        # for j, corner_id in enumerate(all_charuco_ids[i]):
        #     corner_idx = corner_id[0]
        #     matched_obj_points.append(corners_3d[corner_idx])
        #     matched_img_points.append(all_charuco_corners[i][j])
        # if len(matched_obj_points) > 0:
        #     objPoints.append(np.array(matched_obj_points, dtype=np.float32))
        #     imgPoints.append(np.array(matched_img_points, dtype=np.float32))

        matched_obj_points, matched_img_points = board.matchImagePoints(
            all_charuco_corners[i], all_charuco_ids[i]
        )
        if matched_obj_points is not None and len(matched_obj_points) > 0:
            objPoints.append(matched_obj_points)
            imgPoints.append(matched_img_points)
    
    # Calibrate using standard cv2.calibrateCamera (new API for OpenCV 4.7.0+)
    ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        objPoints, imgPoints, imageSize, None, None
    )
    
    print(f"\nCalibration successful with RMS error: {ret}")
    print(f"\nCamera Matrix:\n{cameraMatrix}")
    print(f"\nDistortion Coefficients:\n{distCoeffs}")
    print(f"\nReprojection Error (RMS): {ret}")  # ret is the RMS error
    
    # Save results
    calibration_folder = os.path.join(os.path.dirname(__file__), 'calibration_results')
    os.makedirs(calibration_folder, exist_ok=True)
    
    np.save(os.path.join(calibration_folder, 'camera_matrix.npy'), cameraMatrix)
    np.save(os.path.join(calibration_folder, 'dist_coeffs.npy'), distCoeffs)
    
    print(f"\nCalibration results saved to {calibration_folder}/")
else:
    print("No valid calibration images found!")

