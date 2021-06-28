import csv
import os
import numpy as np

class CaptureLandmarks:
    def __init__(self, _results):
        self.results = _results

    num_coords = len(results.pose_landmarks.landmark)+len(result.face_landmarks.landmark)

    landmarks = ['class']
    for val in range(1, num_coords+1):
        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]

    with open('coords.csv', mode='w', newline='') as f:
        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv.writer.writerow(landmarks)