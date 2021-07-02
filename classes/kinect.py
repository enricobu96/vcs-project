from openni import openni2, nite2, utils, _openni2
import sys
import cv2
import openni
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
import mediapipe as mp



class Kinect:

    def __init__(self):
        openni2.initialize('./lib/openni')
        self.dev = openni2.Device.open_any()

    def initialize_rgbcamera(self):
        color_camera = self.dev.create_color_stream()
        return color_camera

    def initialize_depthcamera(self):
        depth_camera = self.dev.create_depth_stream()
        return depth_camera

    def start_user_tracker(self):

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

        openni2.initialize('./lib/openni')
        dev = openni2.Device.open_any()
        video = dev.create_color_stream()
        # video.set_video_mode(_openni2.OniVideoMode(pixelFormat = _openni2.OniPixelFormat.ONI_PIXEL_FORMAT_RGB888, resolutionX=320, resolutionY=240))

        video.start()
        

        """
        CAPTURING PHASE
        Show webcam with landmarks and save landmarks is coords.csv with class 'gesture'
        """
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
          while True:
            frame = video.read_frame()
            frame_data = frame.get_buffer_as_uint16()
            image = np.frombuffer(frame_data, dtype=np.uint8)
            image.shape = (480,640,3)
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = np.concatenate((image, image, image), axis=0)
            # image = np.swapaxes(image, 0, 2)
            # image = np.swapaxes(image, 0, 1)
            cv2.imshow('Raw Webcam Feed', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break


        # cv2.destroyAllWindows()
        # openni2.unload()

        



# """
# OPENNI
# """
# openni2.initialize('./lib/openni/')
# dev = openni2.Device.open_any()
# depth_stream = dev.create_depth_stream()
# depth_stream.start()
