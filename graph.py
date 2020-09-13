from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt
import brewer2mpl
import numpy as np
import math

def showImg(start, end, X):
    nums = end-start
    qnums = math.floor(nums**0.5)
    fig = plt.figure(figsize=(20,20))
    for i in range(start, end+1):
        input_img = X[i:(i+1),:,:,:]
        ax = fig.add_subplot(qnums+1,qnums+1,i+1)#col, row, location
        ax.imshow(input_img[0,:,:,0], cmap=plt.cm.gray)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    plt.show()


def histogram(y_train,label_name): 
    #convert one-hot to label
    labels  = [np.argmax(lst) for lst in y_train]
    colorset = brewer2mpl.get_map('Set3', 'qualitative', 6).mpl_colors
    #y_train.shape[1] width
    fig = plt.figure(figsize=(8,y_train.shape[1]))
    ax = fig.add_subplot(1,1,1)
    #get histogram
    hist = np.bincount(labels)
    ax.bar(np.arange(0,len(hist)),hist , color=colorset, alpha=0.8)
    ax.set_xticks(np.arange(0,len(hist),1))
    ax.set_xticklabels(label_name, rotation=60, fontsize=14)
    ax.set_xlim([0, len(hist)])
    ax.set_ylim([0, y_train.shape[0]])
    ax.set_title('hist')
    
    plt.tight_layout()
    plt.show()


def modelHistory(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()



def confusionMatrix(x_test, y_test,model,label_name):
    y_prob = model.predict(x_test, batch_size=32, verbose=0)
    y_pred = [np.argmax(prob) for prob in y_prob]
    y_true = [np.argmax(true) for true in y_test]


    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure(figsize=(cm.shape[1],cm.shape[1]))
    matplotlib.rcParams.update({'font.size': 16})
    ax  = fig.add_subplot(1,1,1)
    matrix = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)#plt.cm.YlGnBu
    fig.colorbar(matrix) 
    for i in range(0,cm.shape[0]):
        for j in range(0,cm.shape[1]):  
            ax.text(j,i,cm[i,j],va='center', ha='center')
    ticks = np.arange(cm.shape[0])
    ax.set_xticks(ticks)
    ax.set_xticklabels(label_name, rotation=45)
    ax.set_yticks(ticks)
    ax.set_yticklabels(label_name)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()