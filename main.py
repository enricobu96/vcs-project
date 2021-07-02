__author__ = "E. Buratto, M. Sciacco"
__projectName__ = "CiPenseremoPerIlNome"
__version__ = "0.0.1"
__license__ = "GPLv3"

import argparse
from enum import Enum
from operator import sub

from model.mediapipe.acquire_data_webcam import AcquireData
from model.mediapipe.acquire_data_dataset import AcquireDataset
from model.mediapipe.run import Run
from model.mediapipe.train import Train
from model.kinect_depth.acquire_kinect import AcquireKinect
from time import sleep

class ClassToAcquire(Enum):
      greet = 1
      dab = 2
      tpose = 3
      jazzhands = 4

def main(args):
      
      if args.subcommand == 'acquire':
            if hasattr(ClassToAcquire, args.gesture):
                  a = AcquireData()
                  a.acquire_data(args.gesture)
            else:
                  print('Wrong class')
                  exit
      
      elif args.subcommand == 'acquire-dataset':
            a = AcquireDataset()
            a.acquire_data()
      
      elif args.subcommand == 'train':
            t = Train()
            t.train()

      elif args.subcommand == 'aseqtrain':
            for i in ["greet","greet","dab","dab","tpose","tpose"]:
                  print("NOW DO: ", i)
                  sleep(2)
                  a = AcquireData()
                  a.acquire_data(i)
                  
            print("now im training yea boiiiiii")
            t = Train()
            t.train()      

      elif args.subcommand == 'run':
            r = Run()
            r.run()

      elif args.subcommand == 'acquire-kinect':
            if hasattr(ClassToAcquire, args.gesture):
                  ks = AcquireKinect()
                  ks.acquire_kinect(args.gesture)
            else:
                  print('Wrong class')
                  exit
            

if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      subparsers = parser.add_subparsers(dest='subcommand')
      parser_acquire = subparsers.add_parser('acquire')
      parser_acquire2 = subparsers.add_parser('aseqtrain')
      parser_acquire3 = subparsers.add_parser('acquire-dataset')
      parser_acquire.add_argument('gesture', help='The gesture to acquire')
      parser_run = subparsers.add_parser('run')
      parser_train = subparsers.add_parser('train')
      parser_kinect = subparsers.add_parser('acquire-kinect')
      parser_kinect.add_argument('gesture', help='The gesture to acquire')
      args = parser.parse_args()
      main(args)
