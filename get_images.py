import cv2
import numpy as np
import os


for num in range(1,4,1):
    video = '/home/natalia/openpose/examples/media/Turn_on/Turn_on{}.mp4'.format(num)
    cap = cv2.VideoCapture(video)
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    step = 4
    n = int(total_frames//step)
    os.mkdir("/home/natalia/openpose/examples/media/Turn_on/Turn_on{}".format(num))
    for i in range(n):
        # here, we set the parameter 1 which is the frame number to the frame (i*frames_step)
        cap.set(1, i * step)
        success, image = cap.read()

        path = '/home/natalia/openpose/examples/media/Turn_on/Turn_on{}/image_{}.jpg'.format(num, i)
        # save your image
        cv2.imwrite(path, image)
    cap.release()