import os
from video import get_frames
from model_utils import get_features_batch
from config.global_parameters import default_model_name
from config.resources import video_resource
from glob import glob
import numpy as np
from collections import defaultdict
from sklearn.metrics import confusion_matrix
from keras.models import load_model

def test_video(videoPath):
    frames = list(get_frames(videoPath, time_step=1000))
    if len(frames)==0:
        print("Error in video")
        return
    
    print("Processing",videoPath)
    modelName = "lstmOFvgg16_9g_bs32_ep100.h5"
    model = load_model("/home/parkhi/Desktop/DIP_project/data/models/"+ modelName)
    videoFeatures = get_features_batch(frames)
    videoFeatures = np.array(videoFeatures)
    videoFeatures = np.reshape(videoFeatures, (videoFeatures.shape[0], 1, videoFeatures.shape[1]))
    predictedClasses = model.predict_classes(videoFeatures)
    predictedScores = model.predict(videoFeatures)
    return predictedClasses, predictedScores
if __name__=="__main__":

    from sys import argv
    genres, scores = test_video(argv[1])
    genres = np.reshape(genres,len(genres))
    predictedGenre = np.argmax(np.bincount(genres))                                                  
    genreDict = {0:'Chases',1:'Dance',2:'Eating',3:'Fight',4:'Heated_Discussions',5:'Normal_Chatting',6:'Romance',7:'Running',8:'Tragic'}
    mapping = ['Chases','Dance','Eating','Fight','Heated_Discussions','Normal_Chatting','Romance','Running','Tragic']                                       
    frameSequence=' | '.join([genreDict[key] for key in genres])                                     

    print(mapping[predictedGenre])
    #print(frameSequence)
