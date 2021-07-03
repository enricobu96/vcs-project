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


    def __init__(self, min_subsequence_treshold: int, min_gesture_threshold: float):
        self.frame_buffer = [] # (gesture_name, double precision)
        self.min_subsequence_threshold = min_subsequence_treshold | 20
        self.min_gesture_threshold = min_gesture_threshold
    

    def addToBufferAndCheck(self, g_name: str, g_prec: float):
        #print(len(self.frame_buffer))
        if(len(self.frame_buffer) == 90):
            checker = self.checkBuffer()
            self.frame_buffer.pop()
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

            # Check if gesture is adjacent or nearly adjacent
            if(d[frame[NAME]] in (range(i-5, i))):
                d[frame[NAME]] = i
                c[frame[NAME]] += 1
                p[frame[NAME]] = p[frame[NAME]] + ((frame[PREC] - p[frame[NAME]]) / c[frame[NAME]])
            
            i+=1

        #print(c)
        gesture = max(c.items())
        print(gesture)
        if(gesture[1] >= self.min_subsequence_threshold):
            print("Found", gesture[NAME])
            return gesture[NAME]
        else:
            print("No gesture with enough precision")
            return "no"


def main():
    ga = GestureAssistant(20, 0.75)

    for x in range(180):
        if(x < 30):
            ga.addToBufferAndCheck("greet", 0.76)
        elif(x < 120):
            ga.addToBufferAndCheck("dab", 0.78)
        elif(x < 180):
            ga.addToBufferAndCheck("idle", 0.34)
        elif(x < 260):
            ga.addToBufferAndCheck("tpose", 0.90)

if __name__ == '__main__':
    main()