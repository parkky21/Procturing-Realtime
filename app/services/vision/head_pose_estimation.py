# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 03:00:36 2020

@author: hp
"""

import cv2
import numpy as np

def process_head_pose(img, landmarks):
    """
    Process head pose using MediaPipe landmarks.
    
    MediaPipe Face Mesh Landmarks:
    Nose Tip: 1
    Chin: 152
    Left Eye Left Corner: 33
    Right Eye Right Corner: 263
    Left Mouth Corner: 61
    Right Mouth Corner: 291
    """
    alerts = []
    
    if not landmarks:
        return img, alerts

    h, w, _ = img.shape
    # New API: landmarks is already a list of NormalizedLandmark
    mesh_points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in landmarks])

    # 3D Model Points (Standard Generic Face)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ])
    
    # 2D Image Points from Landmarks
    image_points = np.array([
        mesh_points[1],     # Nose tip
        mesh_points[152],   # Chin
        mesh_points[33],    # Left eye left corner
        mesh_points[263],   # Right eye right corner
        mesh_points[61],    # Left mouth corner
        mesh_points[291]    # Right mouth corner
    ], dtype="double")
    
    # Camera Internals
    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")
    
    dist_coeffs = np.zeros((4,1))
    
    # Solve PnP
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, 
        image_points, 
        camera_matrix, 
        dist_coeffs, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    # Project Nose point to draw direction line
    (nose_end_point2D, jacobian) = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]), 
        rotation_vector, 
        translation_vector, 
        camera_matrix, 
        dist_coeffs
    )
    
    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    cv2.line(img, p1, p2, (255, 0, 0), 2)
    
    # Calculate Euler Angles from Rotation Vector
    rmat, jac = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    
    # angles from RQDecomp3x3 are already in degrees
    x = angles[0] # Pitch
    y = angles[1] # Yaw
    z = angles[2] # Roll
    
    # --- Neutral Pose Calibration ---
    global calibration_frames, neutral_pose, calibration_start_frame
    
    if 'calibration_frames' not in globals():
        calibration_frames = []
    if 'neutral_pose' not in globals():
        neutral_pose = None
    if 'calibration_start_frame' not in globals():
        calibration_start_frame = 0 # Counter for warmup
        
    calibration_start_frame += 1
    WARMUP_FRAMES = 30 # Ignore first 30 frames for stability
    CALIBRATION_LIMIT = 100 
    
    if neutral_pose is None:
        if calibration_start_frame < WARMUP_FRAMES:
             cv2.putText(img, "Initializing Camera...", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
             return img, alerts
             
        calibration_frames.append([x, y, z])
        # Show Calibration Status
        progress = int((len(calibration_frames) / CALIBRATION_LIMIT) * 100)
        cv2.putText(img, f"Calibrating: {progress}%", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(img, "Sit Upright & Look at Screen", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if len(calibration_frames) >= CALIBRATION_LIMIT:
            calibration_frames = np.array(calibration_frames)
            neutral_pose = np.mean(calibration_frames, axis=0) # [avg_x, avg_y, avg_z]
        
        return img, alerts
        
    # Apply Neutral Offset
    pitch = x - neutral_pose[0]
    yaw = y - neutral_pose[1]
    
    # Visual Debugging
    p_txt = "Up" if pitch > 0 else "Down"
    y_txt = "Right" if yaw > 0 else "Left"
    
    cv2.putText(img, f"Pitch: {int(pitch)} [{p_txt}] (Neu: {int(neutral_pose[0])})", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, f"Yaw: {int(yaw)} [{y_txt}] (Neu: {int(neutral_pose[1])})", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Relaxed thresholds significantly to prevent false positives during natural minor movements
    THRESH_YAW = 45 
    THRESH_PITCH = 45
    
    if yaw < -THRESH_YAW:
        alerts.append("Head Left")
    elif yaw > THRESH_YAW:
        alerts.append("Head Right")
        
    # Corrected Logic: Negative Pitch is "Head Down"
    if pitch < -THRESH_PITCH:
        alerts.append("Head Down")
    # elif pitch > THRESH_PITCH:
    #     alerts.append("Head Up")
        
    return img, alerts
