__author__ = "E. Buratto, M. Sciacco"
__projectName__ = "CiPenseremoPerIlNome"
__version__ = "0.0.1"
__license__ = "GPLv3"

import argparse
from enum import Enum

import mediapipe as mp
import cv2
from model.train import Train
from model.run import Run

class ClassToTrain(Enum):
      greet = 1
      sad = 2

def main(args):
      
  if args.subcommand == 'train':
        if hasattr(ClassToTrain, args.gesture):
              t = Train()
              t.train(args.gesture)
        else:
              print('Wrong class')
              exit
  elif args.subcommand == 'run':
        r = Run()
        r.run()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(dest='subcommand')
  parser_train = subparsers.add_parser('train')
  parser_train.add_argument('gesture', help='The gesture to train')
  parser_run = subparsers.add_parser('run')
  args = parser.parse_args()
  main(args)

  # kinect.turnOn()
  # while true:
  #   canIStart = waitToSomethingHappen()
  #   if canIStart:
  #     res = recognize()
  #     if res not finish:
  #       google.cMonDoSomething(res)
  #     else:
  #       kinect.standBy()
