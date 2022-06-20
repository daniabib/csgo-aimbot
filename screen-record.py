import numpy as np
import cv2
import time
import mss

frame_width = 1280
frame_height = 720
frame_rate = 20.0
PATH_TO_MIDDLE = "output.avi"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(PATH_TO_MIDDLE, fourcc, frame_rate, 
                      (frame_width, frame_height))

with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080}

    while "Screen capturing":
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(monitor))
        img = cv2.resize(img, (1280, 720))
        frame = img
        frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        cv2.putText(frame, "FPS: %f" % (1.0 / (time.time() - last_time)),
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out.write(frame)
        cv2.imshow('frame', frame)

        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

# Clean up
out.release()
cv2.destroyAllWindows()