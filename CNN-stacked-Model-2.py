
#STACKED ENSEMBLE MODEL- 2

#Here we have added the stacked CNN-2 model only




model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', activation='elu', input_shape=X_train[0].shape))   #using inline activation function 
model.add(Conv2D(32, (3, 3)))
model.add(MaxPooling2D())
model.add(BatchNormalization(axis=-1))
#model.add(Dropout(0.20))



model.add(Conv2D(64, (3, 3), padding='same', activation='elu'))   #using inline activation function 
model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D())
model.add(BatchNormalization(axis=-1))              #subtracts the mean and divide by standard deviation 
#model.add(Dropout(0.20))

#model.add(Conv2D(64, (3, 3), padding='same', activation='elu'))
#model.add(Conv2D(128, (3, 3), activation='elu'))
#model.add(MaxPooling2D())    #for extracting dominant features , downsampling 
#model.add(Dropout(0.10))

model.add(Conv2D(128, (3, 3), padding='same', activation='elu'))
model.add(BatchNormalization(axis=-1))               #mean , std for last axis 
model.add(Conv2D(512, (3, 3), activation='elu'))
model.add(MaxPooling2D())
#model.add(Dropout(0.25))
model.add(Flatten())                                       #flatteing into i dimensional array 
model.add(Dense(512, activation='elu'))
model.add(Dropout(0.50))
model.add(Dense(activation='sigmoid', units=1))    #densely connected full of neurons so all neurons get resutl from prevous neurons 

model.compile(loss='binary_crossentropy', optimizer = opt, metrics=["accuracy"])

model.summary()     





