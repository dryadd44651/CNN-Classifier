import pandas as pd
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data(sample_split=0.3, usage='Training', data = []):
    data = data[data.Usage == usage]
    rows = random.sample(list(data.index), int(len(data)*sample_split))
    data = data.loc[rows]
    x = list(data["pixels"])
    X = []
    #convert piexels to list
    for i in range(len(x)):
        each_pixel = [int(num) for num in x[i].split()]
        X.append(each_pixel)
    ## reshape into 48*48*1 and rescale
    X = np.array(X)
    # 1 dimension to 3 dimension
    X = np.dstack([X, X, X])
    X = X.reshape(X.shape[0], 48, 48,3)
    X = X.astype("float32")
    X /= 255
    
    #convert to one hot
    y_train = to_categorical(data.emotion)
    #print(y_train)
    return X, y_train

def load_selected(rawData,selected,emotion):
    frames = []
    for s in selected:
        tmp = rawData[rawData['emotion'] == emotion[s]]
        frames.append(tmp)

    newEmo = {}
    mapper = {}
    for i,s in enumerate(selected):
        newEmo[s] = i
        mapper[emotion[s]] = i
    #concat: combines the rows
    data = pd.concat(frames, axis=0)
    data['emotion'] = data['emotion'].map(mapper)
    return data

def loadFace():
    ## All three datasets are well loaded accordingly
    #emo: input the class you want to train
    emotion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,'Sad': 4, 'Surprise': 5, 'Neutral': 6}
    selected     = ['Angry', 'Happy']
    path='fer20131.csv'


    rawData = pd.read_csv(path)

    data = load_selected(rawData,selected,emotion)
    ratio = 0.2

    X_test, y_test = load_data(sample_split=ratio, usage='PrivateTest',data = data)

    X_train, y_train = load_data(sample_split=ratio, usage= 'Training',data = data)
    #validation_data
    X_val,y_val = load_data(sample_split=ratio, usage= 'PublicTest',data = data)
    return X_train, X_val, y_train, y_val,X_test,y_test,selected


def loadDigitalData():

    selected     = ['0','1', '2','3', '4','5', '6','7', '8','9']
    folderbig = "small"
    foldersmall = "small"

    train_x_location = folderbig + "/" + "x_train.csv"
    train_y_location = folderbig + "/" + "y_train.csv"
    test_x_location = foldersmall + "/" + "x_test.csv"
    test_y_location = foldersmall + "/" + "y_test.csv"



    print("Reading training data")
    x_train_2d = pd.read_csv(train_x_location,header = None)
    y_train = pd.read_csv(train_y_location,names = ["label"])


    print("Reading testing data")
    x_test_2d = np.loadtxt(test_x_location, dtype="uint8", delimiter=",")
    x_test_3d = x_test_2d.reshape(-1,28,28,1)
    X_test = x_test_3d
    y_test = np.loadtxt(test_y_location, dtype="uint8", delimiter=",")

    x_train_3d = x_train_2d.values.reshape(-1,28,28,1)
    x_train = x_train_3d

    print("Pre processing x of training data")
    x_train = x_train / 255.0

    y_train = to_categorical(y_train, num_classes = 10)
    y_test = to_categorical(y_test, num_classes = 10)

    random_seed = 2
    # Split the train and the validation set for the fitting
    X_train, X_val, y_train, y_val = train_test_split(x_train, y_train, test_size = 0.8, random_state=random_seed)
    return X_train, X_val, y_train, y_val,X_test,y_test,selected

import scipy as sp
def loadDigitalData3D():

    selected     = ['0','1', '2','3', '4','5', '6','7', '8','9']
    folderbig = "small"
    foldersmall = "small"

    train_x_location = folderbig + "/" + "x_train.csv"
    train_y_location = folderbig + "/" + "y_train.csv"
    test_x_location = foldersmall + "/" + "x_test.csv"
    test_y_location = foldersmall + "/" + "y_test.csv"



    
    #x_train_2d = pd.read_csv(train_x_location,header = None)
    y_train = pd.read_csv(train_y_location,names = ["label"])


    print("Reading testing data")
    #x_test_2d = np.loadtxt(test_x_location, dtype="uint8", delimiter=",")
    x_test_2d = pd.read_csv(test_x_location,header = None).values
    x_test_2d = x_test_2d.reshape(-1,28,28)
    x_test_2d = sp.ndimage.zoom(x_test_2d,(1,1.2,1.2),order=1)
    s = x_test_2d.shape
    x_test_3d = np.dstack([x_test_2d,x_test_2d,x_test_2d])
    X_test = x_test_3d.reshape(-1,s[1],s[2],3)
    y_test = np.loadtxt(test_y_location, dtype="uint8", delimiter=",")
    print("Reading training data")
    #x_train_2d = pd.read_csv(train_x_location,header = None)
    x_train_2d = pd.read_csv(train_x_location,header = None).values
    x_train_2d = x_train_2d.reshape(-1,28,28)
    x_train_2d = sp.ndimage.zoom(x_train_2d,(1,1.2,1.2),order=1)
    x_train_3d = np.dstack([x_train_2d,x_train_2d,x_train_2d])
    s = x_train_2d.shape
    X_train = x_train_3d.reshape(-1,s[1],s[2],3)
    print("Pre processing x of training data")
    X_train = X_train / 255.0

    y_train = to_categorical(y_train, num_classes = 10)
    y_test = to_categorical(y_test, num_classes = 10)

    random_seed = 2
    # Split the train and the validation set for the fitting
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = 0.8, random_state=random_seed)
    return X_train, X_val, y_train, y_val,X_test,y_test,selected

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
def process_data(root,img_dims):

    x_train,y_train = getDataLabel(root,"train",img_dims)
    x_test,y_test = getDataLabel(root, "test",img_dims)
    x_val,y_val = getDataLabel(root,"val",img_dims)

        
    
    return x_train, x_val, y_train, y_val,x_test,y_test,['NORMAL','PNEUMONIA']
    #return x_test, x_val, y_test, y_val,x_test,y_test,['NORMAL','PNEUMONIA']



def getDataLabel(root,path,img_dims):
    data = []
    labels = []
    for cond in ['/NORMAL/', '/PNEUMONIA/']:
        for name in (os.listdir(root + path + cond)):
            try:
                img = cv2.imread(root+ path +cond+name,1)
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (img_dims, img_dims))
                #img = np.dstack([img, img, img])
                img = img.astype('float32') / 255
                if cond=='/NORMAL/':
                    label = 0
                elif cond=='/PNEUMONIA/':
                    label = 1
                data.append(img)
                labels.append(label)
                # cv2.imshow("img", img)
                # cv2.waitKey(0)
            except:
                pass
    data = np.array(data)
    #data = data.reshape(-1,img_dims,img_dims,1)
    data = data.reshape(-1,img_dims,img_dims,3)
    labels = np.array(labels)
    labels = to_categorical(labels)
    return data,labels

def resizeSave(root,img_dims):
    
    folder1 = root+'resize_'+str(img_dims)
    if not os.path.isdir(folder1): os.mkdir(folder1)
    for path in ["train","test","val"]:
        folder2 = folder1+'/'+ path
        if not os.path.isdir(folder2): os.mkdir(folder2)
        data = []
        labels = []
        for cond in ['/NORMAL/', '/PNEUMONIA/']:
            folder3 = folder2+cond
            if not os.path.isdir(folder3): os.mkdir(folder3)
            for name in (os.listdir(root + path + cond)):
                try:
                    img = cv2.imread(root+ path +cond+name,1)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = cv2.resize(img, (img_dims, img_dims))
                    
                    #img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX )
                    img = cv2.equalizeHist(img)
                    cv2.imwrite(root+'resize_'+str(img_dims)+'/'+ path +cond+name, img) 
                except:
                    pass

