import mediapipe as mp
import cv2
import numpy as np
from numpy.lib.type_check import imag
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import pickle
from classes.kinect import Kinect
from time import time

class Run:

    def run(self):
        """
        IMPORT MODEL
        Import model from binary dump
        """
        with open('./model/mediapipe/prediction_models/prediction_model.pkl', 'rb') as f:
            model = pickle.load(f)

        """
        MEDIAPIPE INITIALIZATION
        We use mediapipe holistic in order to recognize every body component we need
        We recognize only body poses, but we use holistic in order to expand the functionalities in the future
        """
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic

        """
        KINECT INITIALIZATION
        Initialize Kinect both rgb and depth camera
        """
        k = Kinect()
        rgb_camera = k.initialize_rgbcamera()
        rgb_camera.start()

        """
        CAPTURING PHASE
        Show webcam with landmarks and do real-time predictions
        """
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            while True:
                """
                RGB CAMERA
                Retrieve rgb camera frame and do image filtering on that
                """
                frame = rgb_camera.read_frame()
                frame_data = frame.get_buffer_as_uint16()
                image = np.frombuffer(frame_data, dtype=np.uint8)
                image.shape = (480,640,3)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                """
                HOLISTIC PROCESS
                Apply holistic process on image
                """
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                
                # Draw pose landmarks
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

                try:
                    # Get pose keypoints
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose], ).flatten())    
                    row = pose_row
                    X = pd.DataFrame([row])

                    # Do predictions
                    gesture_class = model.predict(X)[0]
                    body_language_prob = model.predict_proba(X)[0]

                    # Get ear coordinates (to center the writings)
                    coords = tuple(np.multiply(
                                np.array(
                                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x, 
                                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y))
                            , [640,480]).astype(int))
                
                    cv2.rectangle(image, 
                            (coords[0], coords[1]+5), 
                            (coords[0]+len(gesture_class)*20, coords[1]-30), 
                            (245, 117, 16), -1)
                    cv2.putText(image, gesture_class, coords, 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                    # Get status box
                    cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

                    # Display Class
                    cv2.putText(image, 'CLASS'
                                , (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, gesture_class.split(' ')[0]
                            , (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                    # Display Probability
                    cv2.putText(image, 'PROB'
                                , (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(body_language_prob[np.argmax(body_language_prob)],2))
                                , (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                except:
                    pass

                cv2.imshow('Raw Webcam Feed', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()
        k.close_camera()
