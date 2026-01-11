import cv2
import numpy as np
import mediapipe as mp

def process_eye(img, landmarks):
    """
    Process eye gaze using MediaPipe Iris landmarks.
    
    Landmarks used (approximate mapping):
    Left Eye:
    - Left Corner: 33
    - Right Corner: 133
    - Iris Center: 468
    
    Right Eye:
    - Left Corner: 362
    - Right Corner: 263
    - Iris Center: 473
    """
    alerts = []
    
    if not landmarks:
        return img, alerts

    h, w, _ = img.shape
    # New API: landmarks is already a list of NormalizedLandmark objects
    mesh_points = np.array([np.multiply([p.x, p.y], [w, h]).astype(int) for p in landmarks])

    # Indices for landmarks
    LEFT_IRIS = [474, 475, 476, 477]
    LEFT_PUPIL = 468
    LEFT_EYE_RIGHT_CORNER = 33 
    LEFT_EYE_LEFT_CORNER = 133
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145
    
    RIGHT_IRIS = [469, 470, 471, 472]
    RIGHT_PUPIL = 473
    RIGHT_EYE_RIGHT_CORNER = 362
    RIGHT_EYE_LEFT_CORNER = 263
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374
    
    # --- Blink Detection (False Positive Reduction) ---
    # Calculate Eye Aspect Ratio-ish metric: Height / Width
    
    # Left Eye
    L_top = mesh_points[LEFT_EYE_TOP]
    L_bottom = mesh_points[LEFT_EYE_BOTTOM]
    L_left = mesh_points[LEFT_EYE_LEFT_CORNER]
    L_right = mesh_points[LEFT_EYE_RIGHT_CORNER]
    
    L_height = np.linalg.norm(L_top - L_bottom)
    L_width = np.linalg.norm(L_left - L_right)
    # Avoid division by zero
    L_ratio = L_height / (L_width + 1e-6)
    
    # Right Eye
    R_top = mesh_points[RIGHT_EYE_TOP]
    R_bottom = mesh_points[RIGHT_EYE_BOTTOM]
    R_left = mesh_points[RIGHT_EYE_LEFT_CORNER]
    R_right = mesh_points[RIGHT_EYE_RIGHT_CORNER]
    
    R_height = np.linalg.norm(R_top - R_bottom)
    R_width = np.linalg.norm(R_left - R_right)
    R_ratio = R_height / (R_width + 1e-6)
    
    # Average Opening Ratio
    avg_ratio = (L_ratio + R_ratio) / 2
    
    # If eyes are too closed, don't detect gaze to avoid false positives
    # Relaxed to 0.15 to allow looking down (which narrows eyes)
    if avg_ratio < 0.15:
        # Eyes detected as closed/blinking - skip gaze alert
        return img, alerts

    # --- Gaze Detection ---
    L_pupil = mesh_points[LEFT_PUPIL]
    R_pupil = mesh_points[RIGHT_PUPIL]
    
    # Calculate horizontal ratio (0.0 = Left Corner, 1.0 = Right Corner [physically])
    # Note: On screen (mirrored webcam), User's Left eye:
    # 33 is Inner (Right on screen), 133 is Outer (Left on screen)
    # This directionality can be confusing. Let's stick to vectors.
    
    # User's Left Eye (Right side of screen)
    # Vector from Inner(33) to Outer(133)
    vec_L_width = mesh_points[LEFT_EYE_LEFT_CORNER] - mesh_points[LEFT_EYE_RIGHT_CORNER]
    # Vector from Inner(33) to Pupil
    vec_L_pupil = L_pupil - mesh_points[LEFT_EYE_RIGHT_CORNER]
    
    # Project pupil vector onto width vector to get relative position 0..1
    # Projection = (a . b) / |b|^2
    denom_L = np.sum(vec_L_width**2)
    L_ratio_x = np.sum(vec_L_pupil * vec_L_width) / (denom_L + 1e-6)
    
    # User's Right Eye (Left side of screen)
    # Inner(362) to Outer(263)
    vec_R_width = mesh_points[RIGHT_EYE_LEFT_CORNER] - mesh_points[RIGHT_EYE_RIGHT_CORNER]
    vec_R_pupil = R_pupil - mesh_points[RIGHT_EYE_RIGHT_CORNER]
    denom_R = np.sum(vec_R_width**2)
    R_ratio_x = np.sum(vec_R_pupil * vec_R_width) / (denom_R + 1e-6)
    
    # Average Horizontal Ratio
    # 0.5 is Center
    # > 0.5 means moving towards Outer corner (Looking Left/Right depending on eye)
    # Let's standardize:
    # 33->133 (Inner->Outer left eye): Directions move 'Outwards' (Towards user's left)
    # 362->263 (Inner->Outer right eye): Directions move 'Outwards' (Towards user's right)
    
    # Wait, simple coordinate check:
    # Screen X: 0 (Left) -> Width (Right)
    # User Looking Screen Left (Their Right) -> Pupils move Left (Screen X decreases)
    # User Looking Screen Right (Their Left) -> Pupils move Right (Screen X increases)
    
    # Let's use relative offset from center pupil position again but strictly normalized
    # Center of eye
    L_center_x = (mesh_points[LEFT_EYE_LEFT_CORNER][0] + mesh_points[LEFT_EYE_RIGHT_CORNER][0]) / 2.0
    R_center_x = (mesh_points[RIGHT_EYE_LEFT_CORNER][0] + mesh_points[RIGHT_EYE_RIGHT_CORNER][0]) / 2.0
    
    # Current Pupil X
    L_pupil_x = L_pupil[0]
    R_pupil_x = R_pupil[0]
    
    # Offset from center
    L_off_x = L_pupil_x - L_center_x
    R_off_x = R_pupil_x - R_center_x
    
    # Normalize by eye width
    L_norm_x = L_off_x / L_width
    R_norm_x = R_off_x / R_width
    
    avg_norm_x = (L_norm_x + R_norm_x) / 2.0
    
    # Vertical (Down detection)
    # 0 is Top, Height is Down
    # Eye center Y - USE CORNERS for stability!!
    # Eyelids move, corners are more stable.
    L_center_y = (mesh_points[LEFT_EYE_LEFT_CORNER][1] + mesh_points[LEFT_EYE_RIGHT_CORNER][1]) / 2.0
    R_center_y = (mesh_points[RIGHT_EYE_LEFT_CORNER][1] + mesh_points[RIGHT_EYE_RIGHT_CORNER][1]) / 2.0
    
    L_off_y = L_pupil[1] - L_center_y
    R_off_y = R_pupil[1] - R_center_y
    
    # Normalize by height or width? Width is more stable.
    L_norm_y = L_off_y / L_width
    R_norm_y = R_off_y / R_width 
    
    avg_norm_y = (L_norm_y + R_norm_y) / 2.0
    
    # Thresholds (Lowered for sensitivity)
    # 0.1 means pupil moved 10% of eye width away from center
    THRESH_SIDE = 0.15 
    THRESH_DOWN = 0.03 # Very sensitive for down look, relative to corners
    
    # Looking Right (Screen Left) -> x decreases (negative)
    if avg_norm_x < -THRESH_SIDE:
        alerts.append("Looking Right")
    elif avg_norm_x > THRESH_SIDE:
        alerts.append("Looking Left")
    
    # Looking Down -> y increases (positive)
    if avg_norm_y > THRESH_DOWN:
        alerts.append("Looking Down")

        
    # Draw pupils for visualization
    cv2.circle(img, tuple(L_pupil), 2, (0, 255, 0), -1)
    cv2.circle(img, tuple(R_pupil), 2, (0, 255, 0), -1)

    return img, alerts


