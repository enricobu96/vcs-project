import mediapipe as mp
import cv2
import csv
import os
import numpy as np


class Run:

    def run(self):
        """
        MEDIAPIPE INITIALIZATION
        We use mediapipe holistic in order to recognize every body component we need
        """
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic

        """
        WEBCAM INITIALIZATION
        Initialize webcam and draw landmarks and lines
        """
        cap = cv2.VideoCapture(0)

        """
        CAPTURING PHASE
        Show webcam with landmarks and save landmarks is coords.csv with class 'gesture'
        """
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
          while cap.isOpened():
            _, frame = cap.read()

            # Recolor Feed. We need this bc mp works with RGB but we have BGR
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Make Detections (find keypoints). Results are on: results.face_landmarks, pose_landmarks, left_hand_landmarks and right_hand_landmarks
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            # Back to BGR (from RGB) bc opencv wants BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # DRAW LANDMARKS

            # Draw face landmarks. 468 landmarks
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(80, 110, 10), thickness=1, circle_radius=1),
                                      mp_drawing.DrawingSpec(
                                          color=(80, 256, 121), thickness=1, circle_radius=1)
                                      )

            # Draw right hand landmarks
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(80, 22, 10), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(
                                          color=(80, 44, 121), thickness=2, circle_radius=2)
                                      )

            # Draw left hand landmarks
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(
                                          color=(121, 44, 250), thickness=2, circle_radius=2)
                                      )

            # Draw pose landmarks. 33 landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            cv2.imshow('Raw Webcam Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
              break
        cap.release()
        cv2.destroyAllWindows()
