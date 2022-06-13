import time
from turtle import right

import torch
from mss.linux import MSS as mss
import cv2 as cv
import numpy as np
import pyautogui

# TODO: Entender porque mouse nao funciona direito


class Point:
    def __init__(self, x, y) -> None:
        self.coordinates = np.array([x, y])

    def __sub__(self, other):
        """Calculate euclidean distance between two Point objects."""
        return np.linalg.norm(self.coordinates - other.coordinates)

    def __repr__(self) -> str:
        return f"Point(x={self.coordinates[0]}, y={self.coordinates[1]})"


def calc_target(x, y, w, h) -> Point:
    x_coord = x + w / 2
    y_coord = y + h / 2
    return Point(x_coord.item(), y_coord.item())


def move_mouse(aim_center, target) -> None:
    up = (0, -1)
    up_right = (1, -1)
    right = (1, 0)
    down_right = (1, 1)
    down = (0, 1)
    down_left = (-1, 1)
    left = (-1, 0)

    if aim_center[0] < target[0]:
        _


def on_target(aim_center, target, precision=0.5) -> bool:
    return aim_center - target < precision


AIM_CENTER = Point(x=716, y=428)

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
            if cls == 2:
                pos_to_shoot = calc_target(x, y, w, h)
                # print(pos_to_shoot)
                # print(pyautogui.position())
                # pyautogui.moveTo(pos_to_shoot, duration=1)
                # pyautogui.click()

        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break


# Get bboxes

# Shoot each bbox detected
