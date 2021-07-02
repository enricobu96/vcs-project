from openni import openni2
"""
OPENNI
"""
openni2.initialize('./lib/openni/')
dev = openni2.Device.open_any()
depth_stream = dev.create_depth_stream()
depth_stream.start()
