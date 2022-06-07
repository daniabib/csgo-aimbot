import time

import torch
from mss.linux import MSS as mss
import cv2 as cv
import numpy as np

# Load YOLOv5 with PyTorch Hub from Ultralytics
model = torch.hub.load("ultralytics/yolov5", "custom",
                       path="csgo-detection-v2.pt")

if torch.cuda.is_available():
    model.cuda()

with mss() as sct:
    monitor = {"top": 70, "left": 80, "width": 1280, "height": 720}
    while "Screen Capturing":
        last_time = time.time()

        screenshot = np.array(sct.grab(monitor))

        results = model(screenshot)
        results.render()

        cv.imshow('CV TEST', results.imgs[0])

        print("fps: {}".format(1 / (time.time() - last_time)))

        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break