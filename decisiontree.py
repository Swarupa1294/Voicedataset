import soundfile # to read audio file
import numpy as np
import librosa # to extract speech features
import glob
import os
import pickle # to save model after training
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from features import load_data

emotions_to_observe=['calm', 'happy', 'fearful', 'disgust']

x,y=load_data("C:\\Users\\HP\\Desktop\\DF\\Actor_*\\*.wav",emotions_to_observe)
clf = DecisionTreeClassifier()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=24)
clf.fit(x_train,y_train)

#predict the emotion using testing features
y_pred=clf.predict(x_test)

#calculate accuracy of predictions
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print(f"Accuracy of Model {accuracy*100}")
print(f"Accuracy of Random Guessing {1/len(emotions_to_observe)*100}")

pickle.dump(clf, open("Decisiontree-main.model", "wb"))

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_test,y_pred)
print (matrix)