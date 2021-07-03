import mediapipe as mp
import cv2
import csv
import os
import numpy as np


class AcquireDataset:

    def acquire_data(self):
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
        CAPTURING PHASE
        Get coordinates from images
        """
        filePaths = []
        for root, dirs, files in os.walk('dataset/APE/'):
            for name in files:
                if name.endswith(('.jpg')):
                    print(root + os.sep + name)
                    filePaths.append(root + os.sep + name)

        with mp_holistic.Holistic(static_image_mode=True, model_complexity=2, min_detection_confidence=0.8) as holistic:
            for file in filePaths:
                image = cv2.imread(file)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                try:
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                    row = pose_row

                    class_name = (file.split('/')[2])[0:4]
                    row.insert(0, class_name)
                    with open('./dataset/keypoints/coords_mediapipe.csv', mode='a', newline='') as f:
                        csv_writer = csv.writer(
                            f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(row)
                except Exception as e:
                    print('Error: '+ str(e))
