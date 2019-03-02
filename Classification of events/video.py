import cv2
import numpy as np
import sys,os
from glob import glob
from config.resources import video_resource
from config.global_parameters import frameWidth, frameHeight

def get_frames(videoPath, start_time=5000, end_time=120000, time_step=2000):

    print("Getting frames for ",videoPath)
    try:
        cap = cv2.VideoCapture(videoPath)
        for k in range(start_time, end_time+1, time_step):
            cap.set(cv2.CAP_PROP_POS_MSEC, k)
            success, frame = cap.read()
            if success:
                frame = cv2.resize(frame,(frameWidth, frameHeight))
                yield frame
    except Exception as e:
        print(e)
        return
if __name__=="__main__":
    get_videos(sys.argv[1])
