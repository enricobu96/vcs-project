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
  """
  Mediapipe initialization
  """
  mp_drawing = mp.solutions.drawing_utils
  # We use mediapipe holistic in order to recognize every body component we need
  mp_holistic = mp.solutions.holistic

  """
  Initialize webcam and draw landmarks and lines
  """
  cap = cv2.VideoCapture(0)  # Connect to webcam
  webcam = Webcam(cap)

  # Mediapipe holistic initialization
  with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
      while cap.isOpened():
          ret, frame = cap.read()

          # Recolor Feed. We need this bc mp works with RGB but we have BGR
          image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          image.flags.writeable = False

          # Make Detections (find keypoints). Results are on: results.face_landmarks, pose_landmarks, left_hand_landmarks and right_hand_landmarks
          results = holistic.process(image)

          image.flags.writeable = True
          # Back to BGR (from RGB) bc opencv wants BGR
          image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

          # Draw face landmarks. 468 landmarks
          mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(
                                        color=(80, 110, 10), thickness=1, circle_radius=1),
                                    mp_drawing.DrawingSpec(
                                        color=(80, 256, 121), thickness=1, circle_radius=1)
                                    )

          # Draw right hand landmarks
          mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(
                                        color=(80, 22, 10), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(
                                        color=(80, 44, 121), thickness=2, circle_radius=2)
                                    )

          # Draw left hand landmarks
          mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                    mp_drawing.DrawingSpec(
                                        color=(121, 22, 76), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(
                                        color=(121, 44, 250), thickness=2, circle_radius=2)
                                    )

          # Draw pose landmarks. 33 landmarks
          mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                    mp_drawing.DrawingSpec(
                                        color=(245, 117, 66), thickness=2, circle_radius=4),
                                    mp_drawing.DrawingSpec(
                                        color=(245, 66, 230), thickness=2, circle_radius=2)
                                    )

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
