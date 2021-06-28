__author__ = "E. Buratto, M. Sciacco"
__projectName__ = "CiPenseremoPerIlNome"
__version__ = "0.0.1"
__license__ = "GPLv3"

import argparse

import mediapipe as mp
import cv2
from classes.webcam import Webcam


def main(args):

  # kinect.turnOn()
  # while true:
  #   canIStart = waitToSomethingHappen()
  #   if canIStart:
  #     res = recognize()
  #     if res not finish:
  #       google.cMonDoSomething(res)
  #     else:
  #       kinect.standBy()

  # MEDIAPIPE INITIALIZATION
  mp_drawing = mp.solutions.drawing_utils
  # We use mediapipe holistic in order to recognize every body component we need
  mp_holistic = mp.solutions.holistic

  """
  Initialize webcam and draw landmarks and lines
  """
  cap = cv2.VideoCapture(0)  # Connect to webcam
  webcam = Webcam(mp_drawing, mp_holistic)

  # Mediapipe holistic initialization
  with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
      while cap.isOpened():
          _, frame = cap.read()

          # Recolor Feed. We need this bc mp works with RGB but we have BGR
          image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

          # Make Detections (find keypoints). Results are on: results.face_landmarks, pose_landmarks, left_hand_landmarks and right_hand_landmarks
          image.flags.writeable = False
          results = holistic.process(image)
          image.flags.writeable = True

          # Back to BGR (from RGB) bc opencv wants BGR
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

          webcam.draw_landmarks(image, results)
          webcam.capture_landmarks(results)

          cv2.imshow('Raw Webcam Feed', image)

          if cv2.waitKey(10) & 0xFF == ord('q'):
              break

  cap.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  # TODO: args

  args = parser.parse_args()
  main(args)
