# emotion-detection-from-speech
#### step 1 - Extraxting features from sound
Extracting the mfcc, chroma, and mel features from a sound file. 

Parameters of extract_sound: 
- file_name: filename of the sound file of the spoken sound
- mfcc: Mel Frequency Cepstral Coefficient, represents the short-term power spectrum of a sound
- chroma: Pertains to the 12 different pitch classes
- mel: Mel Spectrogram Frequency

#### Step 2 - define a dictionary 
It holds hold numbers and the emotions available in the RAVDESS dataset.
The labels are found in the naming convention of the dataset
#### Step 3 - define list to hold those emotions we want to observe
#### Step 4: Dataset
-  load the data with a function load_data()
    - Parameter: relative size of the test set
- get all the path names with glob() 
- use pattern “D:\\DataFlair\\ravdess data\\Actor_*\\*.wav”. used in the dataset of RAVADESS
- Emotion labelling by splitting the name around ‘-’ and extracting the third value
#### Step 5: Split the dataset
#### Step 6: Get the shape of the training and testing datasets
#### Step 7: Get the number of features extracted
#### Step 8: Initialize the Multi Layer Perceptron Classifier
This optimizes the log-loss function using LBFGS or stochastic gradient descent. Unlike SVM or Naive Bayes, the MLPClassifier has an internal neural network for the purpose of classification. This is a feedforward ANN model.
#### Step 9: Train the model
#### Step 10: Predict for the test set
#### Step 11: Calculate the accuracy of our model
using accuracy_score() function from sklearn.
## Using the model:
#### Testing with own voice:
#### Step 1: Creating a model and safing it with Pickle
#### Step 2: Extraxting the features out of a .wav file without knowing the emtion 
#### Step 3: Predicting the emotion