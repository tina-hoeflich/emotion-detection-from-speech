
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from training import extract_feature


def load_x(file):
    x,y=[],[]
    # for file in glob.glob("/Users/tina/Documents/emotion-detection-from-speech/test data/own/Tester_*/*.wav"):       
    feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
    x.append(feature)
    return np.array(x)
  
