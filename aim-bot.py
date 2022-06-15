from dataclasses import dataclass
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

@dataclass
class Target:
    x: float
    y: float
    w: float
    h: float 
    confidence: float 
    cls: float

def get_targets(results: list, cls: list[int]):
    """Returns a list with results filtered by the classes specified in the cls argument."""
    return [target for target in results.xywh[0] if target[5] in cls]


def calc_target(target) -> Point:
    x, y, w, h, _, _ = target
    # x_coord = x - w / 2
    # y_coord = y + h / 2
    x_coord = x
    y_coord = y
    return Point(x_coord.item(), y_coord.item())


def on_target(aim_center, target, precision=0.5) -> bool:
    return aim_center - target < precision


def shoot():
    pyautogui.mouseDown()
    time.sleep(0.1)
    pyautogui.mouseUp()


# TODO: Definir função para achar o alvo mais próximo. Melhorar precisão do target.

AIM_CENTER = Point(x=960, y=540)

# Load YOLOv5 with PyTorch Hub from Ultralytics
model = torch.hub.load("ultralytics/yolov5", "custom",
                       path="csgo-detection-v2.pt")

if torch.cuda.is_available():
    model.cuda()

time.sleep(2)

# MAIN LOOP
with mss() as sct:
    # monitor = {"top": 70, "left": 80, "width": 1280, "height": 720}
    while "Screen Capturing":
        last_time = time.time()
        # Grab Screen
        screenshot = np.array(sct.grab(sct.monitors[1]))

        # Prediction
        results = model(screenshot)
        results.render()

        cv.imshow('CV TEST', results.imgs[0])

        print("fps: {}".format(1 / (time.time() - last_time)))

        # Get bboxes
        # for x, y, w, h, confidence, cls in results.xywh[0]:
        #     if (cls == 2 or cls == 3) and confidence > .5:
        #         target = calc_target(x, y, w, h)
        # print(target)
        # while not on_target(AIM_CENTER, target, precision=5):
        #     print(target)

        #     # print("NOT ON TARGET.")
        #     move_mouse(AIM_CENTER, target, speed=3)
        targets = get_targets(results, cls=[2, 3])
        for target in targets:
            target = calc_target(target)
            pyautogui.moveTo(target.x, target.y,
                             #  duration=pyautogui.MINIMUM_DURATION,
                             tween=pyautogui.easeOutQuad)
            if on_target(AIM_CENTER, target, precision=20):
                shoot()

        # print("OUT LOOP")
        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break
