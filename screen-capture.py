from mss.linux import MSS as mss
import cv2 as cv
import numpy as np


with mss() as sct:
    monitor = {"top": 40, "left": 2000, "width": 800, "height": 640}
    while "Screen Capturing":
        screenshot = np.array(sct.grab(monitor))

        cv.imshow('CV TEST', screenshot)

        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break
