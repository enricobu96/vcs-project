
from classes.google import GoogleAPI
import argparse
import zmq
import json
import signal

DEFAULT_GESTURES_JSON_FILE = "config/gestures.json"

signal.signal(signal.SIGINT, signal.SIG_DFL)

class Server:
    def __init__(self) -> None:
        try:
            with open(DEFAULT_GESTURES_JSON_FILE) as f:
                self.gestures = json.load(f)
        except:
            exit("No gestures file found")

    def execute(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.SUB)
        socket.connect('tcp://localhost:6969')
        socket.setsockopt(zmq.SUBSCRIBE, b'')

        while True:
            #  Wait for next request from client
            message = socket.recv_string()
            print(f"Received request: {message}")

            #  Execute request to Google Assistant APIs or reply with error
            if message in self.gestures:
                print("Executing request to Google Assistant -> ", self.gestures[message])
                gapi = GoogleAPI()
                gapi.execute(self.gestures[message])
            else:
                print("The gesture sent (", message, ") has not been recognized!")



def main(args):
    server = Server()
    server.execute()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)