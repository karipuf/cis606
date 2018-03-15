from argparse import ArgumentParser
ap=ArgumentParser()
ap.add_argument('-n',help='Number of epochs (default 60)')
ap.add_argument('-a',help='Architecture of CNNs - defaults to "[32,64,128]"')
ap.add_argument('-l',help='Learning rate to use (default 0.0001)')
ap.add_argument('-d',help='Dropout rate (default 0.2)')
ap.add_argument('-x',help='Width/height of the image (default 128)')
parsed=ap.parse_args()

import numpy as np
import keras
from keras import datasets
from keras.models import Model,Sequential
from keras.layers import Dropout,Dense,Conv2D,Conv2DTranspose,Activation,Input,Convolution2D,Flatten,BatchNormalization,GlobalAveragePooling2D,Softmax,UpSampling2D,MaxPooling2D,AveragePooling2D
from keras.utils import np_utils
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator


####################
# Functions
####################

def ProcessArg(tmp,defaultVal):
        if tmp==None: return defaultVal
        else: return int(tmp)
        

def CreateAutoCNN(nUnits=(32,64,128),inShape=(None,None,3),dropProb=.2,nClasses=10,linearOut=False,learnRate=0.0001):
    
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
    mod.compile(Adam(lr=learnRate),'mse')

    return mod

#######################
# Setting parameters
#######################

nEpochs=ProcessArg(parsed.n,60)
learnRate=ProcessArg(parsed.l,0.0001)
dropProb=ProcessArg(parsed.d,0.2)
imSize=ProcessArg(parsed.x,128)
if parsed.a==None: arch=(32,64,128)
else: arch=eval(parsed.a)    


##################
# Operations
##################

params=zip(('epochs','lr','drop','xy','arch'),(str(tmp) for tmp in [nEpochs,learnRate,dropProb,imSize,arch]))
paramstring='-'.join([tmp[0]+':'+tmp[1] for tmp in params])
print("Params: "+paramstring)

# Prepping the data
imageit=ImageDataGenerator(horizontal_flip=True,rotation_range=20).flow_from_directory("/home/wlwoon/data/LFW/lfw-deepfunneled/",batch_size=15000,target_size=(imSize,imSize))
x=imageit.next()[0]/255.

# Creating model and training
#mod=CreateAutoCNN((32,64,64,128,128))
mod=CreateAutoCNN(nUnits=arch,dropProb=dropProb,learnRate=learnRate)
mod.fit(x,x,epochs=nEpochs,batch_size=128,validation_split=.1)

mod.save('models/facemodel_'+paramstring+'.hdf')
