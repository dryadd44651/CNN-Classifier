

import load
import loadModel
import graph
import pandas as pd
import numpy as np
import sys
import warnings 
from os import path
import tensorflow as tf
warnings.filterwarnings('ignore')


#for face expression
#X_train, X_val, y_train, y_val,X_test, y_test,selected = load.loadFace()

#for digit data
#X_train, X_val, y_train, y_val,X_test, y_test,selected = load.loadDigitalData()

# for ct scan data
#img_dims = 32
#load.resizeSave("./chest_xray/",img_dims)
#X_train, X_val, y_train, y_val,X_test, y_test,selected = load.process_data("./chest_xray/resize_30/",img_dims)
X_train, X_val, y_train, y_val,X_test, y_test,selected = load.loadDigitalData3D()



#graph.showImg(0,50, X_train)

# graph.histogram(y_train,selected)
   

modelName = "model.h5"
historyName = 'history'


#model,history = loadModel.getModel("model","history",X_train,y_train,X_val,y_val,5,128)
#model,history = loadModel.reTrain("model","history",X_train,y_train,X_val,y_val,10,128)
#model,history = loadModel.trainModelResNet50("model","history",X_train,y_train,X_val,y_val,5,128)
model,history = loadModel.trainModelVGG16("model","history",X_train,y_train,X_val,y_val,5,128)

graph.modelHistory(history)




# evaluate model on private test set
score = model.evaluate(X_test, y_test, verbose=0)
print ("model %s: %.2f%%" % (model.metrics_names[1], score[1]*100))

# prediction and true labels



graph.confusionMatrix(X_test, y_test,model,selected)
