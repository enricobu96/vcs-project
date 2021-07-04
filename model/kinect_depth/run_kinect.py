import pickle
from classes.kinect import Kinect, draw_limb, draw_skeleton, get_keypoints, CAPTURE_SIZE_KINECT, CAPTURE_SIZE_OTHERS
from openni import openni2, nite2, utils
import sys
import cv2
import numpy as np
import pandas as pd
import signal
from classes.gesture_assistant import GestureAssistant
import zmq

signal.signal(signal.SIGINT, signal.SIG_DFL)

class RunKinect:
    
    def run(self, classificationModel: str):
        
        """
        IMPORT MODEL
        Import model from binary dump
        """
        fileName = ('./model/kinect_depth/prediction_models/prediction_model_' + classificationModel + '.pkl')
        with open(fileName, 'rb') as f:
            model = pickle.load(f)

        """
        KINECT INITIALIZATION
        Initialize Kinect both rgb and depth camera
        """
        k = Kinect()
        rgb_camera = k.initialize_rgbcamera()
        rgb_camera.start()
        depth_camera = k.initialize_depthcamera()
        depth_camera.start()
        dev_name = k.get_device_info().name.decode('UTF-8')
        use_kinect = False
        if dev_name == 'Kinect':
            use_kinect = True
            print('Using Kinect')

        """
        USER TRACKER INITIALIZATION
        Initialize user tracker and NiTE2 service
        """
        try:
            user_tracker = nite2.UserTracker(k.get_device())
        except utils.NiteError:
            print('Unable to start NiTE human tracker')
            sys.exit(-1)
        (img_w, img_h) = CAPTURE_SIZE_KINECT if use_kinect else CAPTURE_SIZE_OTHERS
        win_w = 640
        win_h = int(img_h * win_w / img_w)

        """
        GESTURE PUBLISHER
        Gesture publisher for Google Assistant
        """
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.bind('tcp://*:6969')
        g_ass = GestureAssistant(5, 60, 20, 0.8, True)

        while True:
            # Get frame
            ut_frame = user_tracker.read_frame()
            depth_frame = ut_frame.get_depth_frame()

            # Image filtering
            depth_frame_data = depth_frame.get_buffer_as_uint16()
            img = np.ndarray((depth_frame.height, depth_frame.width), dtype=np.uint16,
                         buffer=depth_frame_data).astype(np.float32)
            if use_kinect:
                img = img[0:img_h, 0:img_w]
            (min_val, max_val, min_loc, max_loc) = cv2.minMaxLoc(img)
            if (min_val < max_val):
                img = (img - min_val) / (max_val - min_val)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            if ut_frame.users:
                for user in ut_frame.users:
                    if user.is_new():
                        user_tracker.start_skeleton_tracking(user.id)
                    elif (user.state == nite2.UserState.NITE_USER_STATE_VISIBLE and
                        user.skeleton.state == nite2.SkeletonState.NITE_SKELETON_TRACKED):
                        draw_skeleton(img, user_tracker, user, (255, 0, 0))
                        pose_row = list(np.array(get_keypoints(user)).flatten())
                        row = pose_row
                        X = pd.DataFrame([row])

                        # Do predictions
                        if classificationModel == 'lr':
                            gesture_class, gesture_prob = self.__use_lr(model, X, img)
                        elif classificationModel == 'rc':
                            gesture_class, gesture_prob = self.__use_rc(model, X, img)
                        elif classificationModel == 'rf':
                            gesture_class, gesture_prob = self.__use_rf(model, X, img)
                        elif classificationModel == 'gb':
                            gesture_class, gesture_prob = self.__use_gb(model, X, img)
                        elif classificationModel == 'svm':
                            gesture_class, gesture_prob = self.__use_svm(model, X, img)
                        elif classificationModel == 'cnn':
                            gesture_class, gesture_prob = self.__use_cnn(model, X, img)

                        cv2.putText(img, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(img, gesture_class.split(' ')[0], (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(img, 'PROB', (15,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                        cv2.putText(img, str(round(gesture_prob[np.argmax(gesture_prob)],2)), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                        if g_ass.addToBufferAndCheck(gesture_class, gesture_prob[np.argmax(gesture_prob)]):
                            print("sending..")
                        socket.send(bytes(gesture_class,'utf-8')) #(byte?)

                        print(gesture_class, gesture_prob)

                        
            cv2.imshow("Depth", cv2.resize(img, (win_w, win_h)))
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
        k.close_camera()
        cv2.destroyAllWindows()

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
