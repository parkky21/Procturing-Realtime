import cv2
import numpy as np
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

class MediaPipeHandler:
    def __init__(self, model_path="models/face_landmarker.task"):
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True
        )

        self.landmarker = FaceLandmarker.create_from_options(options)

    def process(self, img_bgr):
        """
        Input: BGR image (OpenCV)
        Output: FaceLandmarkerResult
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=img_rgb
        )
        return self.landmarker.detect(mp_image)