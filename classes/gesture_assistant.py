"""
 
 GestureAssistant
 
 ---------------
 This class is used as an adapter to translate the gesture recognized into a google assistant command

 ---------------

"""

NAME = 0 # Gesture name
PREC = 1 # Precision

from time import sleep
from collections import defaultdict
import datetime

class GestureAssistant:

    def __init__(self, cooldown_seconds: int = 0, max_buffer_frames: int = 60, min_repetitions: int = 20, min_precision: float = 0.75, prefer_repetitions_over_precision: bool = True):
        self.frame_buffer = [] # (gesture_name, double precision)
        self.max_buffer_frames = max_buffer_frames
        self.min_repetitions = min_repetitions
        self.min_precision = min_precision
        self.prefer_repetitions = prefer_repetitions_over_precision
        self.cooldown_seconds = cooldown_seconds
        self.__last_cooldown = datetime.datetime.now() - datetime.timedelta(seconds=self.cooldown_seconds)
    

    def addToBufferAndCheck(self, g_name: str, g_prec: float):
        if(self.cooldown_seconds != 0):
            cooldown = datetime.datetime.now() - datetime.timedelta(seconds=self.cooldown_seconds)
            # If last cooldown is bigger than X seconds ago, then the cooldown is still ongoing
            if self.__last_cooldown > cooldown:
                print("cooling down")
                return None

        if(len(self.frame_buffer) == self.max_buffer_frames):
            self.frame_buffer.pop(0)
        checker = self.checkBuffer()
        if(checker != None): 
            self.frame_buffer = []
            self.__last_cooldown = datetime.datetime.now()
            print(checker)
        self.frame_buffer.append((g_name, g_prec))
        return checker

    # Longest common subsequence based on adjacent past 5 items
    def checkBuffer(self):
        
        d = defaultdict(list) # last position
        c = defaultdict(list) # number of repetitions
        p = defaultdict(list) # precision
        i = 0
        for frame in self.frame_buffer:

            # Check if gesture exists
            if(frame[NAME] not in d.keys()):
                d[frame[NAME]] = i
                c[frame[NAME]] = 1
                p[frame[NAME]] = frame[PREC]

            # Check if gesture is adjacent or nearly adjacent
            elif(d[frame[NAME]] in (range(i-5, i))):
                d[frame[NAME]] = i
                c[frame[NAME]] += 1
                p[frame[NAME]] = p[frame[NAME]] + ((frame[PREC] - p[frame[NAME]]) / c[frame[NAME]])
            
            i+=1

        if(i > 0):
            ges = max(c.items())
            alt = max(p.items())

            conditionsGes = ges[1] >= self.min_repetitions and p[ges[NAME]] >= self.min_precision
            conditionsAlt = alt[1] >= self.min_repetitions and p[alt[NAME]] >= self.min_precision

            if conditionsGes and self.prefer_repetitions: 
                return ges[0]
            elif conditionsGes and self.prefer_repetitions == False and conditionsAlt == False:
                return ges[0]
            elif conditionsAlt and self.prefer_repetitions == False:
                return alt[0]
            elif conditionsAlt and self.prefer_repetitions == True and conditionsGes == False:
                return alt[0]
            else:
                return None
        return None


def main():
    ga = GestureAssistant(0, 60, 20, 0.75, True)

    for x in range(260):
        if(x < 30):
            ga.addToBufferAndCheck("greet", 0.76)
        elif(x < 90):
            ga.addToBufferAndCheck("dab", 0.78)
        elif(x < 180):
            ga.addToBufferAndCheck("idle", 0.73)
        elif(x < 260):
            ga.addToBufferAndCheck("tpose", 0.90)

        sleep(0.10)

if __name__ == '__main__':
    main()