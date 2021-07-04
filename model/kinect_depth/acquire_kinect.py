import cv2
import numpy as np
import csv
from classes.kinect import Kinect, draw_limb, draw_skeleton, get_keypoints, CAPTURE_SIZE_KINECT, CAPTURE_SIZE_OTHERS
from openni import openni2, nite2, utils
from time import time
import sys
import os
from time import sleep

class AcquireKinect:

    def acquire_kinect(self, gesture: str):

        if not os.path.isfile('./dataset/keypoints/coords_kinect.csv'):
            print('coords.csv does not exist, creating it...')
            num_coords = 15
            landmarks = ['class']
            for val in range(1, num_coords+1):
                landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val)]

            with open('./dataset/keypoints/coords_kinect.csv', mode='w', newline='') as f:
                csv_writer = csv.writer(
                    f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(landmarks)

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
        COUNTDOWN
        Wait two seconds to get in pose
        """
        print('Ready'), sleep(1), print('Set'), sleep(1), print('Go')


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
        CAPTURING PHASE
        Capture image, do image filtering, apply NiTE2
        """
        countdown = time()
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
                        row.insert(0, gesture)

                        with open('./dataset/keypoints/coords_kinect.csv', mode='a', newline='') as f:
                            csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                            csv_writer.writerow(row)

            cv2.imshow("Depth", cv2.resize(img, (win_w, win_h)))
            if (cv2.waitKey(1) & 0xFF == ord('q')) or time()-countdown >= 10:
                break
        k.close_camera()
        cv2.destroyAllWindows()