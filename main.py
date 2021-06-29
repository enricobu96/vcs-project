__author__ = "E. Buratto, M. Sciacco"
__projectName__ = "CiPenseremoPerIlNome"
__version__ = "0.0.1"
__license__ = "GPLv3"

import argparse
from enum import Enum

from model.acquire_data import AcquireData
from model.run import Run
from model.train import Train

class ClassToAcquire(Enum):
      onlyface = 1
      nothing = 2
      greet = 3
      dab = 4
      tpose = 5


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

      elif args.subcommand == 'run':
            r = Run()
            r.run()

if __name__ == "__main__":
      parser = argparse.ArgumentParser()
      subparsers = parser.add_subparsers(dest='subcommand')
      parser_acquire = subparsers.add_parser('acquire')
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
