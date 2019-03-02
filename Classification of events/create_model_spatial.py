from config.global_parameters import default_model_name
from utils import load_pkl
import numpy as np
from model_utils import spatial_model
import matplotlib.pyplot as plt



def train_classifier(genres=['Chases','Dance','Eating','Fight','Heated_Discussions','Normal_Chatting','Romance','Running','Tragic'], model_name=default_model_name):
    
    trainingData = []
    trainingLabels = []
    num_of_classes = len(genres)
    print("Number of classes:",num_of_classes)
    for genreIndex, genre in enumerate(genres):
        try:
            genreFeatures = load_pkl(genre+"_ultimate_"+default_model_name)
            genreFeatures = np.array([np.array(f) for f in genreFeatures]) # numpy hack
        except Exception as e:
            print(e)
            return
        print("OK.")
        for videoFeatures in genreFeatures:
            randomIndices = range(len(videoFeatures))
            selectedFeatures = np.array(videoFeatures[randomIndices])
            for feature in selectedFeatures:
                trainingData.append(feature)
                trainingLabels.append([genreIndex])
    trainingData = np.array(trainingData)
    trainingLabels = np.array(trainingLabels)
    print("Train data shape : ",trainingData.shape)
    print(trainingLabels.shape)
    model = spatial_model(trainingData)
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    batch_size = 32
    nb_epoch = 100 
    history=model.fit(trainingData, trainingLabels, batch_size=batch_size, nb_epoch=nb_epoch)
    modelOutPath ='data/models/spatial'+model_name+'_'+str(num_of_classes)+"g_bs"+str(batch_size)+"_ep"+str(nb_epoch)+".h5"
    model.save(modelOutPath)
    print("Model saved at",modelOutPath)

    plt.plot(history.history["acc"])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.show()
    
    plt.plot(history.history["loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.show()

if __name__=="__main__":
    train_classifier(genres=['Chases','Dance','Eating','Fight','Heated_Discussions','Normal_Chatting','Romance','Running','Tragic'])
