
# coding: utf-8

# In[ ]:

import numpy as np
from keras import datasets
from keras.models import Model,Sequential
from keras.layers import Dropout,Dense,Conv2D,Conv2DTranspose,Activation,Input,Convolution2D,Flatten,BatchNormalization,GlobalAveragePooling2D,Softmax,UpSampling2D,MaxPooling2D,AveragePooling2D
from keras.utils import np_utils
from keras.optimizers import Adam


# Creating classification network with GAP output
def CreateAutoCNN(nUnits=(32,64,128),inShape=(None,None,3),dropProb=.2,nClasses=10,linearOut=False):
    
    inp=Input(shape=inShape)
    tens=inp
    
    for n in nUnits:
        tens=Conv2D(n,(3,3),padding='valid',strides=2)(tens)
        tens=Dropout(dropProb)(Activation('relu')(BatchNormalization()(tens)))
        
    for n in nUnits[::-1]:
        tens=UpSampling2D()(tens)
        tens=Conv2DTranspose(n,(3,3),padding='valid',strides=1)(tens)
        tens=Dropout(dropProb)(Activation('relu')(BatchNormalization()(tens)))  


    tens=Conv2D(3,(1,1))(tens)
    mod=Model(inputs=inp,outputs=tens)
    mod.compile(Adam(lr=.0001),'mse')

    return mod


# Prepping the data
[(xtrain,ytrain),(xtest,ytest)]=datasets.cifar10.load_data()    
ytrain=np_utils.to_categorical(ytrain)
ytest=np_utils.to_categorical(ytest)
xtrain=xtrain.astype('float32')/255.
xtest=xtest.astype('float32')/255.

# Creating model and training
mod=CreateAutoCNN()
mod.fit(xtrain,xtrain,epochs=20,batch_size=128)

#print(np.sum(np.sum(np.argmax(ytest,axis=1)==np.argmax(mod.predict(xtest),axis=1))))
