import signal
import time
import zmq

signal.signal(signal.SIGINT, signal.SIG_DFL)

context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind('tcp://*:6969')

for i in range(2):
    print("sending..")
    socket.send(b'greet')
    time.sleep(1)