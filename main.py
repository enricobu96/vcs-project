__author__ = "E. Buratto, M. Sciacco"
__projectName__ = "CiPenseremoPerIlNome"
__version__ = "0.0.1"
__license__ = "GPLv3"

import argparse
from enum import Enum

from model.acquire_data import AcquireData
from model.run import Run
from model.train import Train
from time import sleep

class ClassToAcquire(Enum):
      greet = 1
      dab = 2
      tpose = 3

def main(args):
      
      if args.subcommand == 'acquire':
            if hasattr(ClassToAcquire, args.gesture):
                  a = AcquireData()
                  a.acquire_data(args.gesture)
            else:
                  print('Wrong class')
                  exit
      
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

if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      subparsers = parser.add_subparsers(dest='subcommand')
      parser_acquire = subparsers.add_parser('acquire')
      parser_acquire2 = subparsers.add_parser('aseqtrain')
      parser_acquire.add_argument('gesture', help='The gesture to acquire')
      parser_run = subparsers.add_parser('run')
      parser_train = subparsers.add_parser('train')
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
