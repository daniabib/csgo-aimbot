import time
from datetime import datetime
from multiprocessing import Process, Pipe
from multiprocessing.connection import Connection
import numpy as np

import torch
from mss.linux import MSS as mss
import cv2 as cv
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


class Target:
    def __init__(self, target: torch.Tensor) -> None:
        self.x: float = target[0].item()
        self.y: float = target[1].item()
        self.w: float = target[2].item()
        self.h: float = target[3].item()
        self.confidence: float = target[4].item()
        self.label: int = int(target[5].item())
        self.tensor: torch.Tensor = target
        self.coordinates = np.array([self.x, self.y])

    def __repr__(self) -> str:
        return f"Target({self.label}: x={self.x}, y={self.y})"


AIM_CENTER = Point(x=960, y=540)
# displays the frame rate every 2 second
display_time = 1
# set start time to current time
start_time = time.time()
# Set primarry FPS to 0
fps = 0

def get_targets(results: list, labels: list[int]) -> list[Target]:
    """Returns a list with results filtered by the classes specified in the labels argument."""
    return [Target(target) for target in results.xywh[0] if target[5] in labels]


def sort_targets(targets: list[Target], aim_center: Point) -> list[Target]:
    """Sort targets by distance from the aim center."""
    return sorted(targets, key=lambda target: aim_center - target)


def on_target(aim_center, target, precision=0.5) -> bool:
    return aim_center - target < precision


def shoot() -> None:
    pyautogui.mouseDown()
    time.sleep(0.1)
    pyautogui.mouseUp()


def grab_screen(conn: Connection) -> None:
    with mss() as sct:
        while "Screen Capturing":
            screenshot = np.array(sct.grab(sct.monitors[1]))
            conn.send(screenshot)
    # sct = mss()
    # while "Screen Capturing":
    #     scrennshot = np.array(sct.grab(sct.monitors[1]))
    #     conn.send(scrennshot)


def show_screen(conn: Connection) -> None:
    global start_time, fps
    while True:
        results = conn.recv()
        cv.imshow('CV TEST', results)

        # Calculate fps
        fps+=1
        TIME = time.time() - start_time
        if (TIME) >= display_time :
            print("FPS: ", fps / (TIME))
            fps = 0
            start_time = time.time()

        if cv.waitKey(1) == ord('q'):
            cv.destroyAllWindows()
            break


# def save_screen(conn: Connection) -> None:
#     number = 0
#     output = "screenshots/csgo-screenrec-{}.png"
#     to_png = mss.tools.to_png

#     while "there are screenshots":
#         img = queue.get()
#         if img is None:
#             break

#         to_png(img.rgb, img.size, output=output.format(number))
#         number += 1


def run_model(conn1: Connection, conn2: Connection) -> None:
    # Load YOLOv5 from Ultralytics with PyTorch Hub
    model = torch.hub.load("ultralytics/yolov5", "custom",
                           path="csgo-detection-v2.pt")

    if torch.cuda.is_available():
        model.cuda()

    while True:
        # Grab Screen
        screenshot = conn1.recv()

        # Prediction
        results = model(screenshot)
        results.render()

        # Send results
        conn2.send(results.imgs[0])

        targets = get_targets(results, labels=[2, 3])

        if targets:
            target = sort_targets(targets, AIM_CENTER)[0]

            pyautogui.moveTo(target.x, target.y)
            if on_target(AIM_CENTER, target, precision=10):
                shoot()


if __name__ == "__main__":
    # pipe1_output, pipe1_input = Pipe()
    # pipe2_output, pipe2_input = Pipe()

    # p1 = Process(target=grab_screen, args=(pipe1_input,))
    # p2 = Process(target=run_model, args=(pipe1_output, pipe2_input))
    # p3 = Process(target=show_screen, args=(pipe2_output,))

    # p1.start()
    # p2.start()
    # p3.start()
    # Pipes
    p_output, p_input = Pipe()
    p_output2, p_input2 = Pipe()

    # creating new processes
    p1 = Process(target=grab_screen, args=(p_input,))
    p2 = Process(target=run_model, args=(p_output,p_input2,))
    p3 = Process(target=show_screen, args=(p_output2,))

    # starting our processes
    p1.start()
    p2.start()
    p3.start()