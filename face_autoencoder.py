
# coding: utf-8

# In[ ]:

import numpy as np
import keras
from keras import datasets
from keras.models import Model,Sequential
from keras.layers import Dropout,Dense,Conv2D,Conv2DTranspose,Activation,Input,Convolution2D,Flatten,BatchNormalization,GlobalAveragePooling2D,Softmax,UpSampling2D,MaxPooling2D,AveragePooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Creating classification network with GAP output
def CreateAutoCNN(nUnits=(32,64,128),inShape=(None,None,3),dropProb=.2,nClasses=10,linearOut=False):
    
    inp=Input(shape=inShape)
    tens=inp
    
    for n in nUnits:
        tens=Conv2D(n,(3,3),padding='same')(tens)
        tens=MaxPooling2D((2,2),padding='same')(tens)
        tens=Dropout(dropProb)(Activation('relu')(BatchNormalization()(tens)))
        
    for n in nUnits[::-1]:
        tens=Conv2DTranspose(n,(3,3),padding='same',strides=2)(tens)
        #tens=UpSampling2D()(tens)
        tens=Dropout(dropProb)(Activation('relu')(BatchNormalization()(tens)))  


    tens=Conv2D(3,(1,1))(tens)
    mod=Model(inputs=inp,outputs=tens)
    mod.compile(Adam(lr=.0001),'mse')

    return mod


# Prepping the data
#[(xtrain,ytrain),(xtest,ytest)]=datasets.cifar10.load_data()
print("Input images now 128x128, (64,128) CNN")
imageit=ImageDataGenerator(horizontal_flip=True,rotation_range=20).flow_from_directory("/home/wlwoon/data/LFW/lfw-deepfunneled/",batch_size=15000,target_size=(128,128))
x=imageit.next()[0]/255.

# Creating model and training
#mod=CreateAutoCNN((32,64,64,128,128))
mod=CreateAutoCNN((64,128))
mod.fit(x,x,epochs=60,batch_size=128,validation_split=.1)

#print(np.sum(np.sum(np.argmax(ytest,axis=1)==np.argmax(mod.predict(xtest),axis=1))))
