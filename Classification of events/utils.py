from pickle import load, dump
from keras.models import load_model
from config.resources import model_resource

def load_pkl(pklName, verbose=True):
    if verbose:
        print("Loading data from data/{0}.p".format(pklName))
    data = load(open('data/'+pklName+'.p', 'rb'))
    return data


def dump_pkl(data, pklName, verbose = True):

    if verbose:
        print("Dumping data into",pklName)
    dump(data, open('data/'+pklName+'.p', 'wb'))