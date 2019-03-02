import os, sys
from glob import glob
import cv2
import numpy as np
from config.global_parameters import default_model_name
from config.resources import video_resource
from model_utils import get_features_batch
from utils import dump_pkl
from video import get_frames


def gather_training_data(genre, model_name=default_model_name):
    trainPath = os.path.join(video_resource,'train', genre)
    videoPaths = glob(trainPath+'/*')
    genreFeatures = []
    for videoPath in videoPaths:
        print(videoPath,":")
        frames =list(get_frames(videoPath, time_step=1000))
        print(len(frames))
        if len(frames)==0:
            print("corrupt.")
            continue
        videoFeatures = get_features_batch(frames, model_name)
        print(videoFeatures.shape)
        genreFeatures.append(videoFeatures)
    outPath = genre+"_ultimate_"+model_name
    dump_pkl(genreFeatures, outPath)

if __name__=="__main__":
    from sys import argv
    gather_training_data('Chases')
    gather_training_data('Dance')
    gather_training_data('Eating')
    gather_training_data('Fight')
    gather_training_data('Heated_Discussions')
    gather_training_data('Normal_Chatting')
    gather_training_data('Romance')
    gather_training_data('Running')
    gather_training_data('Tragic')
