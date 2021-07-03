__author__ = "E. Buratto, M. Sciacco"
__projectName__ = "CiPenseremoPerIlNome"
__version__ = "0.0.1"
__license__ = "GPLv3"

import argparse
from enum import Enum
from operator import sub

from model.mediapipe.acquire_data_webcam import AcquireData
from model.mediapipe.acquire_data_dataset import AcquireDataset
from model.mediapipe.run_mediapipe import Run
from model.train import Train
from model.kinect_depth.acquire_kinect import AcquireKinect
from model.kinect_depth.run_kinect import RunKinect
from time import sleep

class ClassToAcquire(Enum):
      greet = 1
      dab = 2
      tpose = 3
      jazzhands = 4

class TrainRunModes(Enum):
      mediapipe = 1
      nite = 2

class Classifications(Enum):
      lr = 1
      rc = 2
      rf = 3
      gb = 4
      svm = 5
      cnn = 6

def main(args):

      if args.subcommand == 'acquire-mediapipe':
            if hasattr(ClassToAcquire, args.gesture):
                  a = AcquireData()
                  a.acquire_data(args.gesture)
            else:
                  print('Wrong class')
                  exit
      
      elif args.subcommand == 'acquire-dataset':
            a = AcquireDataset()
            a.acquire_data()

      elif args.subcommand == 'aseqtrain':
            for i in ["greet","greet","dab","dab","tpose","tpose"]:
                  print("NOW DO: ", i)
                  sleep(2)
                  a = AcquireData()
                  a.acquire_data(i)
            t = Train()
            t.train()     
      
      elif args.subcommand == 'acquire-kinect':
            if hasattr(ClassToAcquire, args.gesture):
                  ks = AcquireKinect()
                  ks.acquire_kinect(args.gesture)
            else:
                  print('Wrong class')
                  exit

      elif args.subcommand == 'train':
            if hasattr(TrainRunModes, args.mode):
                  t = Train()
                  t.train(args.mode == 'nite')
            else:
                  print('You came to the wrong neighborood')

      elif args.subcommand == 'run':
            if args.mode == 'mediapipe' and hasattr(Classifications, args.classification):
                  r = Run()
                  r.run(args.classification)
            elif args.mode == 'nite' and hasattr(Classifications, args.classification):
                  r = RunKinect()
                  r.run(args.classification)
            else:
                  print('You came to the wrong neighborood')
            

if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      subparsers = parser.add_subparsers(dest='subcommand')

      parser_acquire_mp = subparsers.add_parser('acquire-mediapipe')
      parser_acquire_ds = subparsers.add_parser('acquire-dataset')
      parser_acquire_kn = subparsers.add_parser('acquire-kinect')
      parser_acquire_seq = subparsers.add_parser('aseqtrain')
      parser_acquire_mp.add_argument('gesture', help='The gesture to acquire')
      parser_acquire_kn.add_argument('gesture', help='The gesture to acquire')

      parser_train = subparsers.add_parser('train')
      parser_train.add_argument('mode', help='Mode to train [mediapipe/nite]')

      parser_run = subparsers.add_parser('run')
      parser_run.add_argument('mode', help='Running mode [mediapipe/nite]')
      parser_run.add_argument('classification', help='Classification algorithm to use')  
      
      args = parser.parse_args()
      main(args)
