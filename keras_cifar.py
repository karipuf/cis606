
# coding: utf-8

# In[ ]:

import numpy as np
from keras import datasets
from keras.models import Model,Sequential
from keras.layers import Dropout,Dense,Conv2D,Activation,Input,Convolution2D,Flatten,BatchNormalization,GlobalAveragePooling2D,Softmax
from keras.utils import np_utils
from keras.optimizers import Adam


# Creating classification network with GAP output
def CreateCNN(nUnits=(32,64,128,128,128,256),inShape=(None,None,3),dropProb=.2,nClasses=10,linearOut=False):
    
    inp=Input(shape=inShape)
    tens=inp
    
    for n in nUnits:
        tens=Conv2D(n,(3,3),padding='valid',strides=(1,1))(tens)
        tens=Dropout(dropProb)(Activation('relu')(BatchNormalization()(tens)))
        
    tens=Conv2D(nClasses,(1,1))(tens)
    tens=GlobalAveragePooling2D()(tens)
    pred=Softmax()(tens)

    #flat=Flatten()(l3)
    #pred=Dense(10,activation='softmax')(flat)

    mod=Model(inputs=inp,outputs=pred)
    mod.compile(Adam(lr=.0001),'categorical_crossentropy',['accuracy'])

    return mod


# Prepping the data
[(xtrain,ytrain),(xtest,ytest)]=datasets.cifar10.load_data()    
ytrain=np_utils.to_categorical(ytrain)
ytest=np_utils.to_categorical(ytest)
xtrain=xtrain.astype('float32')/255.
xtest=xtest.astype('float32')/255.

# Creating model and training
mod=CreateCNN()
mod.fit(xtrain,ytrain,epochs=20,batch_size=128)

print(np.sum(np.sum(np.argmax(ytest,axis=1)==np.argmax(mod.predict(xtest),axis=1))))
