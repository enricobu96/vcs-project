import mediapipe as mp
import cv2
import csv
import os
import numpy as np
from time import time, sleep
from classes.kinect import Kinect
from random import uniform
import sys


class AcquireData:

    def acquire_data(self, gesture: str):
        """
        MEDIAPIPE INITIALIZATION
        We use mediapipe holistic in order to recognize every body component we need
        """
        mp_drawing = mp.solutions.drawing_utils
        mp_holistic = mp.solutions.holistic

        """
        CAPTURE LANDMARKS INITIALIZATION
        Initialize number of coordinates and coords.csv file (if file doesn't exist)
        """
        if not os.path.isfile('./dataset/keypoints/coords_mediapipe.csv'):
            print('coords.csv does not exist, creating it...')
            num_coords = 33 
            landmarks = ['class']
            for val in range(1, num_coords+1):
                landmarks += ['x{}'.format(val), 'y{}'.format(val),
                                'z{}'.format(val), 'v{}'.format(val)]

            with open('./dataset/keypoints/coords_mediapipe.csv', mode='w', newline='') as f:
                csv_writer = csv.writer(
                    f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)

        """
        COUNTDOWN
        Wait two seconds to get in pose
        """
        print('Ready'), sleep(0.25), print('Set'), sleep(0.25), print('Go')

        """
        KINECT INITIALIZATION
        Initialize Kinect both rgb and depth camera
        """
        k = Kinect()
        rgb_camera = k.initialize_rgbcamera()
        depth_camera = k.initialize_depthcamera()
        rgb_camera.start()
        depth_camera.start()

        """
        INITIALIZE SAVE FOR LATER
        Initialize object to save video for later post-processing
        """
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('./model/_vid/temp.avi', fourcc, 20, (640,480))

        """
        CAPTURING PHASE
        Show webcam with landmarks and save landmarks is coords.csv with class 'gesture'
        """
        countdown = time()
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
            SAVE FOR LATER
            Save video for post-processing
            """
            out.write(image)

            """
            HOLISTIC PROCESS
            Apply holistic process on image(s)
            """
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            try:
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())          
                row = pose_row 
                row.insert(0, gesture)

                with open('./dataset/keypoints/coords_mediapipe.csv', mode='a', newline='') as f:
                    csv_writer = csv.writer(
                        f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)
            except Exception as e:
                print('Did not get landmark: '+ str(e))
                pass

            cv2.imshow('Raw Webcam Feed', image)

            if (cv2.waitKey(10) & 0xFF == ord('q')) or time()-countdown >= 10:
              break
        cv2.destroyAllWindows()
        k.close_camera()
        out.release()

        """
        DATA AUGMENTATION
        Use the previous recorded video to do post processing (data augmentation)
        """
        print('Wait, processing...')
        sys.stdout.flush()
        
        cap = cv2.VideoCapture('./model/_vid/temp.avi')
        if not cap.isOpened():
            print('Error opening video file')
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                ret, image = cap.read()
                if ret == True:
                    image_flip, dark_image, bright_image, resized_image = self.__data_augmentation(image)
                    ifresults, diresults, biresults, riresults= holistic.process(image_flip), holistic.process(dark_image), holistic.process(bright_image), holistic.process(resized_image)

                    try:
                        # spaghetti alla carbonara
                        if_pose = ifresults.pose_landmarks.landmark
                        di_pose = diresults.pose_landmarks.landmark
                        bi_pose = biresults.pose_landmarks.landmark
                        ri_pose = riresults.pose_landmarks.landmark
                        if_pose_row = list(np.array([[landmark.x, landmark.y, landmark.z,  landmark.visibility] for landmark in if_pose]).flatten())          
                        di_pose_row = list(np.array([[landmark.x, landmark.y, landmark.z,  landmark.visibility] for landmark in di_pose]).flatten())          
                        bi_pose_row = list(np.array([[landmark.x, landmark.y, landmark.z,  landmark.visibility] for landmark in bi_pose]).flatten())          
                        ri_pose_row = list(np.array([[landmark.x, landmark.y, landmark.z,  landmark.visibility] for landmark in ri_pose]).flatten())          
                        
                        # panna e salmone
                        if_pose_row.insert(0, gesture)
                        di_pose_row.insert(0, gesture)
                        bi_pose_row.insert(0, gesture)
                        ri_pose_row.insert(0, gesture)

                        with open('./dataset/keypoints/coords_mediapipe.csv', mode='a', newline='') as f:
                            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            # carbonara ketchup e wurstel
                            csv_writer.writerow(if_pose_row)
                            csv_writer.writerow(di_pose_row)
                            csv_writer.writerow(bi_pose_row)
                            csv_writer.writerow(ri_pose_row)

                    except Exception as e:
                        print('Did not get landmark: '+ str(e))
                        pass
                else:
                    break
        
        os.remove('./model/_vid/temp.avi')
        
        print('Processing done, kthxbye')
        sys.stdout.flush()

    def __data_augmentation(self, image):
        flip = cv2.flip(image, 1)
        dark = cv2.convertScaleAbs(image, alpha=uniform(0.5, 1), beta=0)
        bright = cv2.convertScaleAbs(image, alpha=uniform(1, 1.5))
        resized = cv2.resize(image, (0,0), fx=uniform(0.9,1), fy=uniform(0.9,1))

        return flip, dark, bright, resized
