"""
 
 GestureAssistant
 
 ---------------
 This class is used as an adapter to translate the gesture recognized into a google assistant command

 ---------------

"""

NAME = 0 # Gesture name
PREC = 1 # Precision

from collections import defaultdict

class GestureAssistant:


    def __init__(self, min_repetitions: int, min_precision: float):
        self.frame_buffer = [] # (gesture_name, double precision)
        self.min_repetitions = min_repetitions | 20
        self.min_precision = min_precision
    

    def addToBufferAndCheck(self, g_name: str, g_prec: float):
        #print(len(self.frame_buffer))
        if(len(self.frame_buffer) == 90):
            # print("its 90")
            self.frame_buffer.pop(0)
        checker = self.checkBuffer()
        if(checker != "no"): self.frame_buffer = []
        self.frame_buffer.append((g_name, g_prec))

    # Longest common subsequence based on average results
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
               # print("Adding", frame[NAME])

            # Check if gesture is adjacent or nearly adjacent
            if(d[frame[NAME]] in (range(i-5, i))):
                d[frame[NAME]] = i
                c[frame[NAME]] += 1
                p[frame[NAME]] = p[frame[NAME]] + ((frame[PREC] - p[frame[NAME]]) / c[frame[NAME]])
            
            i+=1

        if(len(d) > 0):
            #print(c)
            gesture = max(c.items())
            #print(gesture)
            if(gesture[1] >= self.min_repetitions and p[gesture[NAME]] >= self.min_precision):
                print("Found", gesture[NAME])
                return gesture[NAME]
            else:
               # print("No gesture with enough precision")
                return "no"
        return "no"


def main():
    ga = GestureAssistant(20, 0.75)

    g = 0
    for x in range(260):
        if(x < 30):
            ga.addToBufferAndCheck("greet", 0.76)
        elif(x < 120):
            g += 1
            ga.addToBufferAndCheck("dab", 0.78)
        elif(x < 180):
            ga.addToBufferAndCheck("idle", 0.34)
        elif(x < 260):
            ga.addToBufferAndCheck("tpose", 0.90)

if __name__ == '__main__':
    main()