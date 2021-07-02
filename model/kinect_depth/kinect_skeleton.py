import cv2
import numpy as np
from classes.kinect import Kinect, draw_limb, draw_skeleton, CAPTURE_SIZE_KINECT, CAPTURE_SIZE_OTHERS
from openni import openni2, nite2, utils
from time import time
import sys

class KinectSkeleton:

    def run(self):

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
        CAPTURING PHASE
        Capture image, do image filtering, apply NiTE2
        """
        while True:
            ut_frame = user_tracker.read_frame()
            depth_frame = ut_frame.get_depth_frame()
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
                        print("new human id:{} detected.".format(user.id))
                        user_tracker.start_skeleton_tracking(user.id)
                    elif (user.state == nite2.UserState.NITE_USER_STATE_VISIBLE and
                        user.skeleton.state == nite2.SkeletonState.NITE_SKELETON_TRACKED):
                        draw_skeleton(img, user_tracker, user, (255, 0, 0))

            cv2.imshow("Depth", cv2.resize(img, (win_w, win_h)))
            if (cv2.waitKey(1) & 0xFF == ord('q')):
                break
        k.close_camera()
        cv2.destroyAllWindows()





        # """
        # CAPTURING PHASE
        # Show webcam with landmarks and do real-time predictions
        # """
        # while True:
        #     """
        #     RGB CAMERA
        #     Retrieve rgb camera frame and do image filtering on that
        #     """
        #     frame = rgb_camera.read_frame()
        #     frame_data = frame.get_buffer_as_uint16()
        #     image = np.frombuffer(frame_data, dtype=np.uint8)
        #     image.shape = (480,640,3)
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #     """
        #     DEPTH CAMERA
        #     Retrieve depth camera frame and do image filtering on that
        #     """
        #     frame_dep = depth_camera.read_frame()
        #     frame_dep_data = frame_dep.get_buffer_as_uint16()
        #     image_dep = np.frombuffer(frame_dep_data, dtype=np.uint8)
        #     image_dep.shape = (480,640,2)
        #     firstChannel, secondChannel = cv2.split(image_dep)

        #     cv2.imshow('Raw Webcam Feed', image)
        #     if cv2.waitKey(10) & 0xFF == ord('q'):
        #         break
        # cv2.destroyAllWindows()
        # k.close_camera()
