
# coding: utf-8

# In[ ]:

import numpy as np
from keras import datasets
from keras.models import Model,Sequential
from keras.layers import Dropout,Dense,Conv2D,Activation,Input,Convolution2D,Flatten,BatchNormalization
from keras.utils import np_utils
from keras.optimizers import Adam


# In[57]:

[(xtrain,ytrain),(xtest,ytest)]=datasets.cifar10.load_data()


# In[58]:

ytrain=np_utils.to_categorical(ytrain)
ytest=np_utils.to_categorical(ytest)
xtrain=xtrain.astype('float32')/255.
xtest=xtest.astype('float32')/255.


# In[50]:

# Creating classification network
inp=Input(shape=(32,32,3))
l1=Conv2D(32,(3,3),padding='valid',strides=(1,1))(inp)
l1=Dropout(.2)(Activation('relu')(BatchNormalization()(l1)))
l2=Conv2D(64,(3,3),padding='valid',strides=(1,1))(l1)
l2=Dropout(.2)(Activation('relu')(BatchNormalization()(l2)))
l3=Conv2D(128,(3,3),padding='valid',strides=(1,1))(l2)
l3=Dropout(.2)(Activation('relu')(BatchNormalization()(l3)))
l4=Conv2D(128,(3,3),padding='valid',strides=(2,2))(l3)
l4=Dropout(.2)(Activation('relu')(BatchNormalization()(l4)))
l5=Conv2D(128,(3,3),padding='valid',strides=(1,1))(l4)
l5=Dropout(.2)(Activation('relu')(BatchNormalization()(l5)))
flat=Flatten()(l5)
pred=Dense(10,activation='softmax')(flat)
mod=Model(inputs=inp,outputs=pred)
mod.compile(Adam(lr=.0001),'categorical_crossentropy',['accuracy'])


# In[55]:

# Training
#mod.fit(xtrain,ytrain)
mod.fit(xtrain,ytrain,epochs=20,batch_size=128)
print(np.sum(np.sum(np.argmax(ytest,axis=1)==np.argmax(mod.predict(xtest),axis=1))))
