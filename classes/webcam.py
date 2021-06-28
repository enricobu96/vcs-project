import mediapipe as mp
import cv2
import csv
import os
import numpy as np


class Webcam:

    def __init__(self, _mp_drawing, _mp_holistic):
        self.mp_drawing = _mp_drawing
        self.mp_holistic = _mp_holistic

    def draw_landmarks(self, image, results):
        # Draw face landmarks. 468 landmarks
        self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACE_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(
                                           color=(80, 110, 10), thickness=1, circle_radius=1),
                                       self.mp_drawing.DrawingSpec(
                                           color=(80, 256, 121), thickness=1, circle_radius=1)
                                       )

        # Draw right hand landmarks
        self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(
                                           color=(80, 22, 10), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(
                                           color=(80, 44, 121), thickness=2, circle_radius=2)
                                       )

        # Draw left hand landmarks
        self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(
                                           color=(121, 22, 76), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(
                                           color=(121, 44, 250), thickness=2, circle_radius=2)
                                       )

        # Draw pose landmarks. 33 landmarks
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                       self.mp_drawing.DrawingSpec(
                                           color=(245, 117, 66), thickness=2, circle_radius=4),
                                       self.mp_drawing.DrawingSpec(
                                           color=(245, 66, 230), thickness=2, circle_radius=2)
                                       )

    def capture_landmarks(self, results):
        num_coords = len(results.pose_landmarks.landmark) + \
            len(results.face_landmarks.landmark)

        landmarks = ['class']
        for val in range(1, num_coords+1):
            landmarks += ['x{}'.format(val), 'y{}'.format(val),
                          'z{}'.format(val), 'v{}'.format(val)]
        
        # File initialization
        with open('coords.csv', mode='w', newline='') as f:
            csv_writer = csv.writer(
                f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(landmarks)
