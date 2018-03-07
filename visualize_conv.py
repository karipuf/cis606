import keras
import keras.backend as K
import numpy as np
import pylab as pl

#mod=keras.models.load_model('model1.hdf')
mod=keras.models.load_model('gap_model.hdf')

# Inits
outLayer=22
n_iter=100
eta=.8

# Loss function
ims=[]
for filtN in range(16):

    
    f=K.function([mod.layers[0].input],[K.mean(mod.layers[outLayer].output[:,:,:,filtN])])
    grad=K.gradients(f.outputs[0],mod.layers[0].input)[0]
    df=K.function([mod.layers[0].input],[grad])
    im=np.random.rand(1,32,32,3)

    for count in range(n_iter):

        if count%500==0:
            print("Updating iteration #"+str(count)+", loss is "+str(f([im])[0]))

        update=df([im])[0]
        #update=update/(np.sqrt(np.mean(np.square(update)))+.01)
    
        im=im+eta*update
        im=(im-im.min())/(im.max()-im.min())

    ims.append(im[0])

for count in range(16):
    pl.subplot(4,4,count+1)
    pl.imshow(ims[count])


# vgg

if False:
    vgg=keras.applications.VGG16()
    df=K.function([vgg.layers[0].input],K.gradients(K.max(vgg.layers[4].output),vgg.layers[0].input))
    f=K.function([vgg.layers[0].input],[K.max(vgg.layers[4].output)]) # loss

    for count in range(1000):
        if count%500==0:
            print("Updating iteration #"+str(count)+", loss is "+str(f([im])[0]))
        update=df([im])[0]
        update=update/(np.sqrt(np.mean(np.square(update)))+.01)
        
        im=im+.1*update
        im=keras.applications.vgg16.preprocess_input(im)
  
