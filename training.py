#!/usr/bin/env python
# coding: utf-8

# In[3]:


import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# #### step 1 - Extraxting features from sound
# Extracting the mfcc, chroma, and mel features from a sound file. 
# 
# Parameters of extract_sound: 
# - file_name: filename of the sound file of the spoken sound
# - mfcc: Mel Frequency Cepstral Coefficient, represents the short-term power spectrum of a sound
# - chroma: Pertains to the 12 different pitch classes
# - mel: Mel Spectrogram Frequency

# In[4]:


#DataFlair - Extract features (mfcc, chroma, mel) from a sound file
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        if chroma:
            stft=np.abs(librosa.stft(X))
        result=np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result


# #### Step 2 - define a dictionary 
# It holds hold numbers and the emotions available in the RAVDESS dataset.
# The labels are found in the naming convention of the dataset

# In[5]:


emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}


# #### Step 3 - define list to hold those emotions we want to observe
# 

# In[6]:


observed_emotions=['calm', 'happy', 'sad', 'angry','surprised']


# #### Step 4: Dataset
# -  load the data with a function load_data()
#     - Parameter: relative size of the test set
# - get all the path names with glob() 
# - use pattern “D:\\DataFlair\\ravdess data\\Actor_*\\*.wav”. used in the dataset of RAVADESS
# - Emotion labelling by splitting the name around ‘-’ and extracting the third value

# In[13]:


def load_data(test_size=0.1):
    x,y=[],[]
    for file in glob.glob("/Users/tina/Documents/emotion-detection-from-speech/ravdess-data/Actor_*/*.wav"):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
        x.append(feature)
        y.append(emotion)
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


# #### Step 5: Split the dataset
# 

# In[14]:


x_train,x_test,y_train,y_test=load_data(test_size=0.1)


# #### Step 6: Get the shape of the training and testing datasets

# In[15]:


print((x_train.shape[0], x_test.shape[0]))


# #### Step 7: Get the number of features extracted

# In[16]:


print(f'Features extracted: {x_train.shape[1]}')


# #### Step 8: Initialize the Multi Layer Perceptron Classifier
# This optimizes the log-loss function using LBFGS or stochastic gradient descent. Unlike SVM or Naive Bayes, the MLPClassifier has an internal neural network for the purpose of classification. This is a feedforward ANN model.

# In[17]:


model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=700)


# #### Step 9: Train the model and safe it with pickle
# 

# In[18]:


model_trained = model.fit(x_train,y_train)
pickle.dump(model_trained, open("./our_model.pkl", "wb"))


# #### Step 10: Predict for the test set

# In[19]:


y_pred=model.predict(x_test)
for i in y_pred:
    print(i)


# #### Step 11: Calculate the accuracy of our model
# using accuracy_score() function from sklearn.
# 

# In[20]:


accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))

""" 
# #### Testing with own voice:
# extraxting the feature out of a .wav file without knowing the emotion

# In[ ]:


def load_x(file):
    x,y=[],[]
    # for file in glob.glob("/Users/tina/Documents/emotion-detection-from-speech/test data/own/Tester_*/*.wav"):       
    feature=extract_feature(file, mfcc=True, chroma=True, mel=True)
    x.append(feature)
    return np.array(x)
  


# Predicting the emotion

# In[ ]:



x_unknown = load_x()
y_unknown_pred = model.predict(x_unknown)
for i in y_unknown_pred:
  print(i)

 """