from openni import openni2, nite2, utils, _openni2

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

    def close_camera(self):
        openni2.unload()
