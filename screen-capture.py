import time
from datetime import datetime

from PIL import Image

from mss.linux import MSS as mss
from mss.screenshot import ScreenShot


def compress_image(mss_image: ScreenShot, size: tuple[int, int] = None) -> Image:
    image = Image.frombytes("RGB", mss_image.size,
                            mss_image.bgra, "raw", "BGRX")
    if size:
        image = image.resize(size, resample=Image.Resampling.LANCZOS)
    image = image.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
    return image


def capture_screen(fps: float = 1) -> None:
    with mss() as sct:
        sct.compression_level = 7
        monitor = sct.monitors[1]

        while "Screen Capturing":
            timestamp = datetime.now().strftime("%Y%m%d_%H-%M-%S")

            screenshot = sct.grab(monitor)

            # image = compress_image(screenshot, size=(1280, 720))
            image = compress_image(screenshot)
            image.save(f"screenshots/csgo-screenshot-{timestamp}.png")

            time.sleep(1/fps)

if __name__ == "__main__":
    capture_screen()