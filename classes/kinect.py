from openni import openni2, nite2, utils

class Kinect:

    def __init__(self):
        openni2.initialize('./lib/openni')
        nite2.initialize('./lib/nite2')
        self.dev = openni2.Device.open_any()
        self.depth_stream = dev.create_depth_stream()

        

"""
OPENNI
"""
openni2.initialize('./lib/openni/')
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.start()
