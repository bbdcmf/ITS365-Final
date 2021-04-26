#############################################################################
#Import libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
import os, os.path
from os import environ
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#############################################################################
#Suppress matplotlib warnings

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

#############################################################################
#Open image files

def readImg(img):
    image = cv2.imread(img, 0) #0 to signal its grayscale
    imgArray = np.array(image, dtype='float')
    imgArray = np.around(imgArray / 255, 3) #Get brightness of each pixel on a scale of 0.00->1.00 rounded to the 3rd digit.
    return imgArray

##------------------------------------------------------------------------###

#Append the image data to a list

def getImgData(imgList, path):
    for directory in path: #for directories in path
        for f in os.listdir(directory): #for files in each directory
            imgList.append(readImg(os.path.join(directory,f))) 

#############################################################################
#Create the x train data from the images within the path directories

xTrainList = []
path = ["NumbersDataSet/Zero/", "NumbersDataSet/One/", "NumbersDataSet/Two", "NumbersDataSet/Three", "NumbersDataSet/Four", "NumbersDataSet/Five", "NumbersDataSet/Six", "NumbersDataSet/Seven", "NumbersDataSet/Eight", "NumbersDataSet/Nine"]
getImgData(xTrainList, path)
xTrain = np.array(xTrainList)

#############################################################################
#Create the x test data

xTestList = []
path = ["NumbersDataSet/TestSet/"]
getImgData(xTestList, path)
xTest = np.array(xTestList)
xTestReshaped = xTest.reshape(1,-1)

#############################################################################
#Save array to csv file

xTrainReshaped = xTrain.reshape(xTrain.shape[0], -1)
np.savetxt('data1.csv', xTrainReshaped, delimiter='::',  fmt="%.3f")

#############################################################################
#Y labels

y_train = np.array([
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 

2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 

3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,  3,  3,  3,  3,  3, 

4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 

5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,  5,  5,  5,  5,  5,   

6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 

7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 
 
8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 

9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9
])

y_test = np.array([5, 2, 8, 0, 4, 7, 1, 9, 3, 6])

#############################################################################

suppress_qt_warnings()

def accuracy(y_true, y_pred):
    accuracy = np.sum( y_true == y_pred ) / len(y_true)
    return accuracy

#############################################################################
#Keras

model=keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28])) #if flatten is applied to layer having input shape as (batch_size, 2,2), then the output shape of the layer will be (batch_size, 4)
model.add(keras.layers.Dense(units=250, activation='relu')) #relu much quicker than sigmoid and more accurate, but also creates the hockey-stick like graph
model.add(keras.layers.Dense(units=100, activation='relu'))
model.add(keras.layers.Dense(units=50, activation='relu'))
model.add(keras.layers.Dense(units=16, activation='relu'))
model.add(keras.layers.Dense(units=10, activation='softmax')) #softmax: each components will be from 0-1 and the components will add up to 1

model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['SparseCategoricalAccuracy'])  #SGD = Gardient Descent, SparseCategoricalAccuracy is used when labels are ints
hist=model.fit(xTrain, y_train, epochs=200, batch_size=20, validation_data=(xTest, y_test))
score = model.evaluate(xTest, y_test) #Returns (loss, accuracy)
print("Test sample loss: ", score[0], "\nTest sample accuracy: ", score[1])

#############################################################################
#Logistic Regression:

print('\n----------Logistic Regression----------\n')

lr = LogisticRegression(solver = 'saga', max_iter=1000)  #saga used when there are multiple classes
lr.fit(xTrainReshaped, y_train)

lr_pred = []
for i in range(10):
    sample = xTest[i].reshape(1, -1)
    lr_pred.append(int(lr.predict(sample)))
    print(i,'Prediction:',lr_pred[i],'- Actual:',y_test[i])
accuracy_score = accuracy(y_test.astype(int), lr_pred)
print("\nAccuracy: ", accuracy_score*100, '%')
print("Precision Score: ", precision_score(y_test, lr_pred, average='macro', zero_division=0)*100,'%')
print("Recall Score: ", recall_score(y_test, lr_pred, average='macro')*100,'%')
print("F1 Score: ", f1_score(y_test, lr_pred, average='macro')*100,'%')
cmtx = pd.DataFrame(
    confusion_matrix(y_test, lr_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 
    index=['true:0', 'true:1', 'true:2', 'true:3', 'true:4', 'true:5', 'true:6', 'true:7', 'true:8', 'true:9'], 
    columns=['pred:0', 'pred:1', 'pred:2', 'pred:3', 'pred:4', 'pred:5', 'pred:6', 'pred:7', 'pred:8', 'pred:9'],
)
print("Confusion Matrix:")
print(cmtx)
#############################################################################
#KNN:

def euclidean_distance(v1, v2):
    return np.sqrt(np.sum(     (v1 - v2) ** 2    ))
    
#--------------------------------------------------------------------------#

k = 5
def predict_KNN(test_x, i):
    ## calculate distances between test_x and all data samples in X
    distances = [ euclidean_distance(test_x, x )  for x in xTrain   ]
    
    ## sort by distance and return the k closest neighbors
    ## argsort returns the indices of the k nearest neighbors
    neighbors = np.argsort(distances)[:k]
    #print("Neighbor indeces: ", neighbors)
    
    ## extract labels from y_train
    labels = [y_train[i] for i in neighbors]
    print(i, ":", "Labels of most similar indeces: ", labels)
    
    ##select the most common label in labels
    most_common_label = Counter(labels).most_common()
    return most_common_label[0][0], most_common_label[0][1]

#--------------------------------------------------------------------------#

print('\n----------------KNN----------------\n')

knn_pred = []
for i in range(10):
    a, n = predict_KNN(xTest[i], i)
    print(i, ":","Most common label", a, "appearing", n, "out of", k, "times\n")
    knn_pred.append(a)

accuracy_score = accuracy(y_test, knn_pred)
print("Accuracy: ", accuracy_score*100, '%')
print("Precision Score: ", precision_score(y_test, knn_pred, average='macro')*100,'%')
print("Recall Score: ", recall_score(y_test, knn_pred, average='macro')*100,'%')
print("F1 Score: ", f1_score(y_test, knn_pred, average='macro')*100,'%')

cmtx = pd.DataFrame(
    confusion_matrix(y_test, knn_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]), 
    index=['true:0', 'true:1', 'true:2', 'true:3', 'true:4', 'true:5', 'true:6', 'true:7', 'true:8', 'true:9'], 
    columns=['pred:0', 'pred:1', 'pred:2', 'pred:3', 'pred:4', 'pred:5', 'pred:6', 'pred:7', 'pred:8', 'pred:9'],
)
print("Confusion Matrix:")
print(cmtx)
print('\n-----------------------------------\n');

#############################################################################
#Show the xTest data as an image

fig = plt.figure(figsize=(10, 7))
rows = 2
columns = 5
plt.title('Test Sample Images:')
plt.axis('off') #disable unnecessary graphics
for i in range(10):
    fig.add_subplot(rows, columns, i+1) #function starts at 1 so add 1 to i
    plt.imshow(xTest[i])
    plt.axis('off')
    plt.title(i+1)
print("#################DONE###################")

plt.show()

#############################################################################
