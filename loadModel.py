from os import path
from keras.models import model_from_json
import keras
import pickle
from keras import layers
from keras import models
from keras.models import Model
from keras import optimizers
from keras.layers import Flatten, Dense
from keras.applications.resnet50 import ResNet50
class History_trained_model(object):
    def __init__(self, history, epoch, params):
        self.history = history
        self.epoch = epoch
        self.params = params

def getModel(modelName,historyName,x_train,y_train,x_validation,y_validation,nb_epoch,batch_size):
    if path.exists(modelName+'.h5') and path.exists(modelName+'.json') and path.exists(historyName):
        weight = loadWeight(modelName)
        history = loadHistory(historyName)
        
    else:
        weight,history = trainModel(modelName,historyName,x_train,y_train,x_validation,y_validation,nb_epoch,batch_size)
        
    return weight,history


def loadWeight(modelName):
    print("Load the model from",modelName)
    json_file = open(modelName+'.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    
    # load weights into new model
    model.load_weights(modelName+'.h5')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
    

def loadHistory(historyName):
    with open(historyName, 'rb') as file:
        history=pickle.load(file)
    return history



def trainModel(modelName,historyName,x_train,y_train,x_validation,y_validation,nb_epoch,batch_size):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu',
                            input_shape=(x_train.shape[1], x_train.shape[2], x_train.shape[-1])))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(y_train.shape[1], activation='softmax'))#softmax
    #sigmoid is better for binary class
    # optimizer:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print ('Training....')
    from keras.callbacks import ReduceLROnPlateau
    #learning rule
    learning_rate_function = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, #val_acc not improve 3 times in rows
                                            verbose=1, 
                                            factor=0.5, #learing rate *0.5
                                            min_lr=0.00001)# minimum learing rate

    #fit
    history = model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size, callbacks=[learning_rate_function],
            validation_data=(x_validation, y_validation), shuffle=True, verbose=1)
    
        #serialize model to JSON
    model_json = model.to_json()
    with open(modelName+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(modelName+".h5")
    print("Saved model to disk")
    with open(historyName, 'wb') as file:
        history= History_trained_model(history.history, history.epoch, history.params)
        pickle.dump(history, file, pickle.HIGHEST_PROTOCOL)
    return model,history



def reTrain(modelName,historyName,x_train,y_train,x_validation,y_validation,nb_epoch,batch_size):
    if path.exists(modelName+'.h5') and path.exists(modelName+'.json') and path.exists(historyName):
        weight = loadWeight(modelName)
        history = weight.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size,
            validation_data=(x_validation, y_validation), shuffle=True, verbose=1)
        
    else:
        weight,history = trainModel(modelName,historyName,x_train,y_train,x_validation,y_validation,nb_epoch,batch_size)
        
    return weight,history



def trainModelResNet50(modelName,historyName,x_train,y_train,x_validation,y_validation,nb_epoch,batch_size):
    #weights='imagenet' use the pretrain weight
    resnet = ResNet50(input_shape = (x_train.shape[1], x_train.shape[2], 3) ,include_top=False,  weights='imagenet')
    x = Flatten()(resnet.output)
    x = Dense(y_train.shape[1], activation='softmax')(x)
    model = Model(resnet.input, x)
    #sigmoid is better for binary class
    # optimizer:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print ('Training....')
    from keras.callbacks import ReduceLROnPlateau
    #learning rule
    learning_rate_function = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, #val_acc not improve 3 times in rows
                                            verbose=1, 
                                            factor=0.5, #learing rate *0.5
                                            min_lr=0.00001)# minimum learing rate

    #fit
    history = model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size, callbacks=[learning_rate_function],
            validation_data=(x_validation, y_validation), shuffle=True, verbose=1)
    
        #serialize model to JSON
    model_json = model.to_json()
    with open(modelName+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(modelName+".h5")
    print("Saved model to disk")
    with open(historyName, 'wb') as file:
        history= History_trained_model(history.history, history.epoch, history.params)
        pickle.dump(history, file, pickle.HIGHEST_PROTOCOL)
    return model,history

def trainModelVGG16(modelName,historyName,x_train,y_train,x_validation,y_validation,nb_epoch,batch_size):
    #weights='imagenet' use the pretrain weight
    vgg = keras.applications.vgg16.VGG16(input_shape = (x_train.shape[1], x_train.shape[2], 3) ,include_top=False,  weights='imagenet')
    x = Flatten()(vgg.output)
    x = Dense(y_train.shape[1], activation='softmax')(x)
    model = Model(gg.input, x)

    #if you don't want to train the VGG weight
    #VGG weight will be just feature extractor
    #only train the last two layer(linear regrassion)
    # for layer in vgg.layers:
    #     layer.trainable = False

    
    #sigmoid is better for binary class
    # optimizer:
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print ('Training....')
    from keras.callbacks import ReduceLROnPlateau
    #learning rule
    learning_rate_function = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, #val_acc not improve 3 times in rows
                                            verbose=1, 
                                            factor=0.5, #learing rate *0.5
                                            min_lr=0.00001)# minimum learing rate

    #fit
    history = model.fit(x_train, y_train, epochs=nb_epoch, batch_size=batch_size, callbacks=[learning_rate_function],
            validation_data=(x_validation, y_validation), shuffle=True, verbose=1)
    
        #serialize model to JSON
    model_json = model.to_json()
    with open(modelName+".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(modelName+".h5")
    print("Saved model to disk")
    with open(historyName, 'wb') as file:
        history= History_trained_model(history.history, history.epoch, history.params)
        pickle.dump(history, file, pickle.HIGHEST_PROTOCOL)
    return model,history