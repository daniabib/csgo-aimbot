from gc import callbacks
import time
from datetime import datetime

from PIL import Image

from mss.linux import MSS as mss
from mss.screenshot import ScreenShot
import mss.tools as msstol

def compress_image(mss_image: ScreenShot, size: tuple[int, int] = None, resample = None) -> Image:
    image = Image.frombytes("RGB", mss_image.size, mss_image.bgra, "raw", "BGRX")
    # image = image.resize(size, resample=Image.Resampling.LANCZOS)
    image = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    return image


with mss() as sct:
    sct.compression_level = 7
    monitor = sct.monitors[1]

    while "Screen Capturing":
        print(type(Image.Resampling.LANCZOS))
        timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")

        screenshot = sct.grab(monitor)

        image = compress_image(screenshot, size=(1280, 720))  
        image.save("screencaps/test3.png")
        # msstol.to_png(screenshot.rgb,
        #               screenshot.size,
        #             #   output=f"screencaps/csgo-{timestamp}.png",)
        #             output="screencaps/test{date}.png")
        time.sleep(1)
