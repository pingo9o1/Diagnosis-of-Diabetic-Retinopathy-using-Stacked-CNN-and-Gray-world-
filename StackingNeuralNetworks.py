
#Saved Model Weights of different models are combined and a stacking model is prepared using Generalisation technique

def load_all_models(n_models):
    
	all_models = list()
    
	for i in range(n_models):
        
		
		filename = 'models/model_' + str(i + 1) + '.h5'

		model = load_model(filename)
        
		# add to list of members
		all_models.append(model)
		print('>loaded %s' % filename)
        
	return all_models




def define_stacked_model(members):
    
    for i in range(len(members)):
        model=members[i]
        
        for layer in model.layers:                  #layer of model will not train 
            layer.trainable=False
            
            layer._name='ensemble_'+ str(i+1) + '_' + layer.name
            
    
    ensemble_visible= [model.input for model in members]
    
    ensemble_outputs=[model.output for model in members]
    
    merge=concatenate(ensemble_outputs)                         #Model Architretures Merged
    hidden = Dense(64, activation='relu')(merge)                #Final Dense Layer for stacking and combining best weights 
    output = Dense(1, activation='sigmoid')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    
    plot_model(model, show_shapes=True, to_file='model_graph.png')              //To Plot the Stacked model Diagram 
    
    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
    
    return model

        

def fit_stacked_model(model, inputX, inputy):
    
    X=[inputX for _ in range(len(model.input))]
    
    #input_enc=to_categorical(inputy)
    
    model.fit(X, inputy, epochs=20, verbose=0)
    
 
def predict_stacked_model(model , inputX):
    
    X=[inputX for _ in range(len(model.input))]
    
    return model.predict(X, verbose=1)



n_members = 3                                         #3 Different model weights are consodered 
members = load_all_models(n_members)

stacked_model = define_stacked_model(members)

fit_stacked_model(stacked_model, X_train, y_train)



yhat=predict_stacked_model(stacked_model, X_train)                #prediction
acc = accuracy_score(y_train, yhat)

i=0

for i in range(yhat.shape[0]):              #for converting probabilty in label 0 or 1 
    if yhat[i]>0.50:
        yhat[i]=1
    else:
        yhat[i]=0
    



#ypred_class = model.predict_classes(y_hat, verbose=0)

members[1].summary()                  #sanity check for different model memebers 
members[2].summary()
members[0].summary()

print(acc)                            #accuracy 


