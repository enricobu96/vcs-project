import mediapipe as mp
import cv2
import csv
import os
import numpy as np
from time import time, sleep
from classes.kinect import Kinect


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
        if not os.path.isfile('./dataset/keypoints/coords.csv'):
            print('coords.csv does not exist, creating it...')
            num_coords = 33 
            landmarks = ['class']
            for val in range(1, num_coords+1):
                landmarks += ['x{}'.format(val), 'y{}'.format(val),
                                'z{}'.format(val), 'v{}'.format(val)]

            with open('./dataset/keypoints/coords.csv', mode='w', newline='') as f:
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
            HOLISTIC PROCESS
            Apply holistic process on image
            """
            image.flags.writeable = False
            results = holistic.process(image)
            image.flags.writeable = True

            # Draw pose landmarks. 33 landmarks
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(
                                          color=(245, 117, 66), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(
                                          color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            try:
                # Get pose keypoints
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())          
                row = pose_row 
                row.insert(0, gesture)
                
                # Save found keypoints on coords.csv
                with open('./dataset/keypoints/coords.csv', mode='a', newline='') as f:
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
