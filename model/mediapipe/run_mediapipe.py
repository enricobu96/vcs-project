import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
import pickle
from classes.kinect import Kinect
import signal
from classes.gesture_assistant import GestureAssistant
import zmq

signal.signal(signal.SIGINT, signal.SIG_DFL)


class Run:

    def run(self, classificationModel: str):
        """
        IMPORT MODEL
        Import model from binary dump
        """
        fileName = ('./model/mediapipe/prediction_models/prediction_model_' + classificationModel + '.pkl')
        with open(fileName, 'rb') as f:
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

            """
            GESTURE PUBLISHER
            Gesture publisher for Google Assistant
            """
            context = zmq.Context()
            socket = context.socket(zmq.PUB)
            socket.bind('tcp://*:6969')
            g_ass = GestureAssistant(5, 60, 20, 0.8, True)

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
                    if classificationModel == 'lr':
                        gesture_class, gesture_prob = self.__use_lr(model, X, image)
                    elif classificationModel == 'rc':
                        gesture_class, gesture_prob = self.__use_rc(model, X, image)
                    elif classificationModel == 'rf':
                        gesture_class, gesture_prob = self.__use_rf(model, X, image)
                    elif classificationModel == 'gb':
                        gesture_class, gesture_prob = self.__use_gb(model, X, image)
                    elif classificationModel == 'svm':
                        gesture_class, gesture_prob = self.__use_svm(model, X, image)
                    elif classificationModel == 'cnn':
                        gesture_class, gesture_prob = self.__use_cnn(model, X, image)

                    if g_ass.addToBufferAndCheck(gesture_class, gesture_prob[np.argmax(gesture_prob)]):
                        print("sending..")
                        socket.send(bytes(gesture_class,'utf-8')) #(byte?)

                    cv2.putText(image, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, gesture_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    cv2.putText(image, str(round(gesture_prob[np.argmax(gesture_prob)],2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    print(gesture_class, gesture_prob)
                
                except:
                    pass

                cv2.imshow('Raw Webcam Feed', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
        cv2.destroyAllWindows()
        k.close_camera()

    def __use_lr(self, model, X, img):
        gesture_class, gesture_prob = model.predict(X)[0], model.predict_proba(X)[0]
        return gesture_class, gesture_prob

    def __use_rc(self, model, X, img):
        gesture_class = model.predict(X)[0]
        d = model.decision_function(X)[0]
        gesture_prob = np.exp(d)/np.sum(np.exp(d))
        return gesture_class, gesture_prob

    def __use_rf(self, model, X, img):
        gesture_class, gesture_prob = model.predict(X)[0], model.predict_proba(X)[0]
        return gesture_class, gesture_prob

    def __use_gb(self, model, X, img):
        gesture_class, gesture_prob = model.predict(X)[0], model.predict_proba(X)[0]
        return gesture_class, gesture_prob

    def __use_svm(self, model, X, img):
        gesture_class, gesture_prob = model.predict(X)[0], model.predict_proba(X)[0]
        return gesture_class, gesture_prob

    def __use_cnn(self, model, X, img):
        gesture_class, gesture_prob = model.predict(X)[0], model.predict_proba(X)[0]
        return gesture_class, gesture_prob