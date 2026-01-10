import cv2
import mediapipe as mp
import numpy as np

class MediaPipeHandler:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        # Refine landmarks=True gives us Iris landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process(self, img):
        """
        Processes an image and returns the face landmarks.
        Input: BGR image (OpenCV)
        Output: results object from MediaPipe
        """
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        return results
