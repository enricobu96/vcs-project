import mediapipe as mp
import cv2
import csv
import os
import sys
import numpy as np
import openni
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
# 4 diversi modelli di classificazione
# TODO fare un confronto di questi in termini di precision and recall
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle
from classes.kinect import Kinect

class Run:

    def run(self):
        # KINECT MODE
        k = Kinect()
        k.start_user_tracker()

        """
        IMPORT MODEL
        """
        with open('./model/prediction_models/prediction_model.pkl', 'rb') as f:
            model = pickle.load(f)

        """
        MEDIAPIPE INITIALIZATION
        We use mediapipe holistic in order to recognize every body component we need
        """
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic


        # """
        # WEBCAM INITIALIZATION
        # Initialize webcam and draw landmarks and lines
        # """
        # cap = cv2.VideoCapture(0)

        # """
        # CAPTURING PHASE
        # Show webcam with landmarks and save landmarks is coords.csv with class 'gesture'
        # """
        # with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        #   while cap.isOpened():
        #     ret, frame = cap.read()

        #     # Recolor Feed. We need this bc mp works with RGB but we have BGR
        #     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     resized_image = cv2.resize(image, (640,480), interpolation=cv2.INTER_AREA)

        #     # Make Detections (find keypoints). Results are on: results.face_landmarks, pose_landmarks, left_hand_landmarks and right_hand_landmarks
        #     resized_image.flags.writeable = False
        #     results = holistic.process(resized_image)
        #     resized_image.flags.writeable = True

        #     # Back to BGR (from RGB) bc opencv wants BGR
        #     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        #     # Draw pose landmarks. 33 landmarks
        #     mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        #                               mp_drawing.DrawingSpec(
        #                                   color=(245, 117, 66), thickness=2, circle_radius=4),
        #                               mp_drawing.DrawingSpec(
        #                                   color=(245, 66, 230), thickness=2, circle_radius=2)
        #                               )

        #     try:
        #         pose = results.pose_landmarks.landmark
        #         pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose], ).flatten())      
                
        #         row = pose_row

        #         # Make Detections
        #         X = pd.DataFrame([row])
        #         gesture_class = model.predict(X)[0]
        #         body_language_prob = model.predict_proba(X)[0]
        #         #print(gesture_class, body_language_prob)
                
        #         # Grab ear coords
        #         coords = tuple(np.multiply(
        #                         np.array(
        #                             (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
        #                             results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
        #                     , [640,480]).astype(int))
                
        #         cv2.rectangle(image, 
        #                     (coords[0], coords[1]+5), 
        #                     (coords[0]+len(gesture_class)*20, coords[1]-30), 
        #                     (245, 117, 16), -1)
        #         cv2.putText(image, gesture_class, coords, 
        #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
        #         # Get status box
        #         cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)
                
        #         # Display Class
        #         cv2.putText(image, 'CLASS'
        #                     , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        #         cv2.putText(image, gesture_class.split(' ')[0]
        #                     , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
        #         # Display Probability
        #         cv2.putText(image, 'PROB'
        #                     , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        #         cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
        #                     , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
        #     except:
        #         pass
            
        #     cv2.imshow('Raw Webcam Feed', image)

        #     if cv2.waitKey(10) & 0xFF == ord('q'):
        #       break
        # cap.release()
        # cv2.destroyAllWindows()
