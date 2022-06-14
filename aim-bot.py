import time

import torch
from mss.linux import MSS as mss
import cv2 as cv
import numpy as np
import pyautogui



class Point:
    def __init__(self, x, y) -> None:
        self.x = x
        self.y = y
        self.coordinates = np.array([x, y])

    def __sub__(self, other) -> float:
        """Calculates euclidean distance between two Points."""
        return np.linalg.norm(self.coordinates - other.coordinates)

    def __repr__(self) -> str:
        return f"Point(x={self.x}, y={self.y})"


def calc_target(x, y, w, h) -> Point:
    x_coord = x + w / 2
    y_coord = y + h / 2
    return Point(x_coord.item(), y_coord.item())


def move_mouse(aim_center, target, speed=1) -> None:
    UP = (0, -speed)
    UP_RIGHT = (speed, -speed)
    RIGHT = (speed, 0)
    DOWN_RIGHT = (speed, speed)
    DOWN = (0, speed)
    DOWN_LEFT = (-speed, speed)
    LEFT = (-speed, 0)
    UP_LEFT = (-speed, -speed)

    if aim_center.x < target.x and aim_center.y < target.y:
        pyautogui.move(DOWN_RIGHT)
    elif aim_center.x < target.x and aim_center.y > target.y:
        pyautogui.move(UP_RIGHT)
    elif aim_center.x > target.x and aim_center.y < target.y:
        pyautogui.move(DOWN_LEFT)
    elif aim_center.x > target.x and aim_center.y > target.y:
        pyautogui.move(UP_LEFT)
    elif aim_center.x == target.x and aim_center.y < target.y:
        pyautogui.move(DOWN)
    elif aim_center.x == target.x and aim_center.y > target.y:
        pyautogui.move(UP)
    elif aim_center.x < target.x and aim_center.y == target.y:
        pyautogui.move(RIGHT)
    elif aim_center.x > target.x and aim_center.y == target.y:
        pyautogui.move(LEFT)
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
        # TEST_TARGET = Point(x=1580, y=400)


        # Get bboxes
        for x, y, w, h, confidence, cls in results.xywh[0]:
            if cls == 0 or cls == 1:
                target = calc_target(x, y, w, h)
                print(target)
                while not on_target(AIM_CENTER, target):
                    print(target)

                    # print("NOT ON TARGET.")
                    # move_mouse(AIM_CENTER, target, speed=3)
                    pyautogui.moveTo(target.x, target.y)
                pyautogui.click()

        print("OUT LOOP")
        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break
