


#test score calculation

scoreTrain=model.evaluate(X_train,y_train,verbose=0)
print("train score: " ,scoreTrain[0])
print("train accuracy :", scoreTrain[1])
scores = model.evaluate(X_test, y_test , verbose=0)
print(' test score :'  ,scores[0])                                                          #evaluation of loss function
print('test accuracy : ' , scores[1])
                                     
print(model.predict_classes(X_test[5:11]))                                         #predicting on test samples
print(y_test[5:11])                                                                 #for cross checking 



"""
Plotting Loss curve and Learning Curves

"""

import matplotlib.pyplot as plt                                                         

plt.plot(history.history['loss'], label = 'train loss')
plt.ylabel('Loss', fontsize=16)
plt.xlabel('Epochs', fontsize=16)
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()


import matplotlib.pyplot as plt

plt.plot(history.history['acc'], label = 'train acc')
plt.ylabel('Accuracy', fontsize=16)
plt.xlabel('Epochs', fontsize=16)
plt.plot(history.history['val_acc'], label='val_acc')
plt.legend()
plt.show()

