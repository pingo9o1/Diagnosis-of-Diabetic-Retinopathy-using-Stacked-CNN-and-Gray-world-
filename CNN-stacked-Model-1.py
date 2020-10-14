

#importing libraries 


import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator, img_to_array  
from keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer    #for converting multi class label to binary labels , so highes tprob gets that class 
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import os
import cv2

import random
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD




#path to our dataset
dataset = 'D:/Pingo/Project/saved'



IMAGE_DIMENSION = (96,96,3)     #3 channel rgb image 
BATCHSIZE = 32                  #hardcoded it below          
data = []                       #for parsing the image with labels , list
classes = []                    #considering our 2 classes (Healthy & Unhealthy)

imagepaths = sorted(list(paths.list_images(dataset)))       #creates sorted list of paths in terminal 
random.seed(42)
random.shuffle(imagepaths)   #shuffling 

for imgpath in imagepaths:
    try:
        image = cv2.imread(imgpath)                     #reading image at the path
        image = cv2.resize(image, (96, 96))             #resized image             
        image_array = img_to_array(image)               #convert image to array  
        data.append(image_array)                        #appended to list 
        label = imgpath.split(os.path.sep)[-2]          #well, splitting it from second last name in path ( basically on basis of folder name)
        classes.append(label)                           #adding the split name that is the label into label list 
    except Exception as e:                              
        print(e)                                        #if any image is not readable 

data = np.array(data,dtype='float')/255.0
labels = np.array(classes)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)                   #converted to binary labels



img_rows,img_cols= 200,200 

data=np.asarray(data)                                   #converts input to array 
classes=np.asarray(labels)

from sklearn.utils import shuffle
Data,Label= shuffle(data,classes, random_state=2)
train_data=[Data,Label]                                 #first in train is data, second is label
type(train_data)



learning_rate=0.00009                                   #hardcoded below 
#batch_size to train
batch_size = 30
# number of output classes
nb_classes = 2
# number of epochs to train
nb_epoch = 20

opt= SGD(lr=learning_rate, decay= learning_rate/ 100)
# number of convolutional filters to use
#nb_filters = 32                                  #Hardcoded it during Model tuning 
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3
(X, y) = (train_data[0],train_data[1])        



from sklearn.model_selection import train_test_split

# STEP A: splitted X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=4)        #Data divided into 80% Train Set and 20% Test Set

print(X_train.shape)                    //Sanity Check - 1
print(X_test.shape)                     //Sanity Check - 2


X_train = X_train.reshape(X_train.shape[0], 96,96, 3)      #no of image samples, width , height channel
X_test = X_test.reshape(X_test.shape[0], 96,96,3)

X_train = X_train.astype('float32')                        #Image Type converted to float type
X_test = X_test.astype('float32')

X_train /= 255                                             #reduced the size for faster convergence or computation 
X_test /= 255


print('X_train shape:', X_train.shape)                    #Sanity Check - 3
print(X_train.shape[0], 'train samples')                  #Sanity Check - 4
print(X_test.shape[0], 'test samples')                    #Sanity Check - 5




//Model ARCHITECTURE - 1 Using Keras API:  


from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization     
from keras.layers.convolutional import Conv2D, MaxPooling2D   
from keras.layers.core import Activation, Flatten,Dense , Dropout    
from keras import backend as k    


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='elu', input_shape=X_train[0].shape))   #using inline activation function 
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D())
model.add(BatchNormalization(axis=-1))
#model.add(Dropout(0.20))                   #Tuned Hyper-parameter



model.add(Conv2D(32, (3, 3), padding='same', activation='elu'))   #using inline activation function 
model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D())
model.add(BatchNormalization(axis=-1))                             #subtracts the mean and divide by standard deviation 
#model.add(Dropout(0.20))                                          #Tuned Hyper-parameter

model.add(Conv2D(64, (3, 3), padding='same', activation='elu'))
model.add(Conv2D(128, (3, 3), activation='elu'))
model.add(MaxPooling2D())                                        #for extracting dominant features , downsampling 
model.add(Dropout(0.10))

model.add(Conv2D(128, (3, 3), padding='same', activation='elu'))
model.add(BatchNormalization(axis=-1))                              
model.add(Conv2D(512, (3, 3), activation='elu'))
model.add(MaxPooling2D())
#model.add(Dropout(0.25))
model.add(Flatten())                                                      #flatteing into 1-dimensional array 
model.add(Dense(512, activation='elu'))
model.add(Dropout(0.50))
model.add(Dense(activation='sigmoid', units=1))                        #densely connected full of neurons so all neurons get resutl from prevous neurons 

model.compile(loss='binary_crossentropy', optimizer = opt, metrics=["accuracy"])              #loss type is Binary Crossentropy 
model.summary()



#Data Augmentation Carried out Below:


from keras.preprocessing.image import ImageDataGenerator

# create generators  - training data will be augmented images
validationdatagenerator = ImageDataGenerator()
traindatagenerator = ImageDataGenerator(width_shift_range=0.1,height_shift_range=0.1,rotation_range=15,
                                        zoom_range=0.1, horizontal_flip=True, fill_mode='nearest' )

batchsize=16            #Hyper-Parameter


#Call Back Regularisation 

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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


train_generator=traindatagenerator.flow(X_train, y_train, batch_size=batchsize, shuffle=True) 
validation_generator=validationdatagenerator.flow(X_test, y_test, batch_size=batchsize, shuffle=True)


#Model Fitting

history=model.fit_generator(train_generator, steps_per_epoch=int(len(X_train)/batchsize), epochs=20, 
                    validation_data=validation_generator, validation_steps=int(len(X_test)/batch_size)
                    )
                    
                    
                    
#Saving Model Weights 
model.save('Desktop/cnn.h5')                     
                
                    
                    
