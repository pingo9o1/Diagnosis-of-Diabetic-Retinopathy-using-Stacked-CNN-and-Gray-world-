

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, Xception, DenseNet201 ,ResNet50, InceptionV3        #pretrained model 
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os
import random
from sklearn.utils import shuffle
from skimage.restoration import denoise_nl_means, estimate_sigma
from tensorflow.keras import backend as K
import bm3d

dataset = 'D:/Pingo/converted/dataset2/testing2'


lr= 0.00009        #learning rate

print("load the images ")

imagepaths= sorted(list(paths.list_images(dataset)))
random.seed(42)

data=[]
classe=[]             #labels of Binary Classification 


for imagepath in imagepaths:

    label=imagepath.split(os.path.sep)[-2]             #label capture 
    classe.append(label)
    image=cv2.imread(imagepath)                        
    image=cv2.resize(image, (224,224))
    data.append(image)

data=np.array(data)/255.0
classes=np.array(classe)

#data_rgb= np.ndarray(shape=(data.shape[0], data.shape[1], data.shape[2], 3), dtype= np.uint8)        #For having custom size images for ResNet and VGG-16

lb=LabelBinarizer()                                        #binary labels now , one hots 
classes=lb.fit_transform(classes)
classes=to_categorical(classes)



data,classes= shuffle(data,classes, random_state=2)

#opt= SGD(lr=0.0009, decay=0.0009/epochs, momentum=0.9)

opt=SGD(lr=lr)




#splitting
(x_train, x_test, y_train, y_test)= train_test_split(data, classes, test_size=0.20, random_state=42)

print(x_train.shape)                      #Sanity check for shapes 
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


x_train = x_train.reshape(x_train.shape[0], 224,224, 3)      #no of image samples, width , height channel
x_test = x_test.reshape(x_test.shape[0], 224,224,3)

#x_train=x_train/255
#x_test=x_test/255


traindatagenerator = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,rotation_range=15,
                                        zoom_range=0.1, horizontal_flip=True, fill_mode='nearest'
                                        )

validationdatagenerator = ImageDataGenerator()  #no change 


#transfer learning 


VGG19_model= VGG16(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))

#ResNet50_model= ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224,224,3)))      #similar weights could be extracted for ResNet50 and further changes can be done in place of VGG16

model=VGG19_model.output
model=AveragePooling2D(pool_size= (2,2))(model)               #pretrained model customization 
model=Flatten(name='flatten')(model)
model=Dense(64, activation='relu')(model)
model=Dropout(0.3)(model)
model=Dense(2, activation= "sigmoid")(model)

#now placing 

model= Model(inputs=VGG19_model.input, outputs=model)             


for l in VGG19_model.layers:
    l.trainable=False                  #freezing layers which dont have to be trained  


print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=["accuracy"])

train_generator= traindatagenerator.flow( x_train , y_train , batch_size = batch_size
                                         ,shuffle=True)

validation_generator= validationdatagenerator.flow(x_test, y_test,batch_size=batch_size, shuffle=True)

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

earlystop= EarlyStopping(monitor= 'val_loss',
                         min_delta=0,
                         patience=2,
                         verbose=1, 
                         restore_best_weights=True)


reduce_lr= ReduceLROnPlateau(monitor= 'val_loss',
                             factor= 0.2,
                             patience=2,
                             verbose=1,
                             min_delta=0.0001)


callbackss= [earlystop, reduce_lr]





history2=model.fit_generator(train_generator, steps_per_epoch=int(len(x_train)/batch_size), epochs=20, 
                    validation_data=validation_generator, validation_steps=int(len(x_test)/batch_size), 
                    callbacks=callbackss)


model.save('Desktop/VGG16.h5')        #or resNet50.h5
