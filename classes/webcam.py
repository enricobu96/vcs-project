import mediapipe as mp
import cv2

class Webcam:
    def __init__(self, _cap):
        self.cap = _cap