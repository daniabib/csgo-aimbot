from dataclasses import dataclass
import time
from turtle import pos

import torch
from mss.linux import MSS as mss
import cv2 as cv
import numpy as np
import pyautogui

from utils import Point

def calc_shoot(x, y, w, h) -> Point:
    x_coord = x + w / 2
    y_coord = y + h / 2
    return Point(int(x_coord.item()), int(y_coord.item()))

def move_mouse():
    ...

def on_target():
    ...

MIDDLE_SCREEN = Point(x=716, y=428)

# Load YOLOv5 with PyTorch Hub from Ultralytics
model = torch.hub.load("ultralytics/yolov5", "custom",
                       path="csgo-detection-v2.pt")

if torch.cuda.is_available():
    model.cuda()

time.sleep(2)

# MAIN LOOP
with mss() as sct:
    monitor = {"top": 70, "left": 80, "width": 1280, "height": 720}
    while "Screen Capturing":
        last_time = time.time()
        # Grab Screen
        screenshot = np.array(sct.grab(monitor))

        # Prediction
        results = model(screenshot)
        results.render()

        cv.imshow('CV TEST', results.imgs[0])

        # print("fps: {}".format(1 / (time.time() - last_time)))

        # Get bboxes
        for x, y, w, h, confidence, cls in results.xywh[0]:
            if cls == 3:
                pos_to_shoot = calc_shoot(x, y, w, h)
                print(pos_to_shoot)
                # print(pyautogui.position())
                # pyautogui.moveTo(pos_to_shoot, duration=1)
                # pyautogui.click()

        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break


# Get bboxes

# Shoot each bbox detected
