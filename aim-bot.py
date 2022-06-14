import time

import torch
from mss.linux import MSS as mss
import cv2 as cv
import numpy as np
import pyautogui



class Point:
    def __init__(self, x, y) -> None:
        self.coordinates = np.array([x, y])

    def __sub__(self, other) -> float:
        """Calculates euclidean distance between two Points."""
        return np.linalg.norm(self.coordinates - other.coordinates)

    def __repr__(self) -> str:
        return f"Point(x={self.coordinates[0]}, y={self.coordinates[1]})"


def calc_target(x, y, w, h) -> Point:
    x_coord = x + w / 2
    y_coord = y + h / 2
    return Point(x_coord.item(), y_coord.item())


def move_mouse(aim_center, target) -> None:
    UP = (0, -1)
    UP_RIGHT = (1, -1)
    RIGHT = (1, 0)
    DOWN_RIGHT = (1, 1)
    DOWN = (0, 1)
    DOWN_LEFT = (-1, 1)
    LEFT = (-1, 0)
    UP_LEFT = (-1, -1)

    if aim_center.x < target.x and aim_center.y < target.y:
        pyautogui.moveRel(DOWN_RIGHT)
    elif aim_center.x < target.x and aim_center.y > target.y:
        pyautogui.moveRel(UP_RIGHT)
    elif aim_center.x > target.x and aim_center.y < target.y:
        pyautogui.moveRel(DOWN_LEFT)
    elif aim_center.x > target.x and aim_center.y > target.y:
        pyautogui.moveRel(UP_LEFT)
    elif aim_center.x == target.x and aim_center.y < target.y:
        pyautogui.moveRel(DOWN)
    elif aim_center.x == target.x and aim_center.y > target.y:
        pyautogui.moveRel(UP)
    elif aim_center.x < target.x and aim_center.y == target.y:
        pyautogui.moveRel(RIGHT)
    elif aim_center.x > target.x and aim_center.y == target.y:
        pyautogui.moveRel(LEFT)
    else:
        print("AT SAME POSITION")

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

        # TEST
        TEST_TARGET = Point(x=1580, y=400)


        # Get bboxes
        for x, y, w, h, confidence, cls in results.xywh[0]:
            if cls == 2:
                target = calc_target(x, y, w, h)
                if on_target(AIM_CENTER, target):
                    pyautogui.rightClick()
                else:
                    print("NOT ON TARGET.")
                    move_mouse(AIM_CENTER, target)
                # print(pos_to_shoot)
                # print(pyautogui.position())
                # pyautogui.moveTo(pos_to_shoot, duration=1)
                # pyautogui.click()

        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break


# Get bboxes

# Shoot each bbox detected
