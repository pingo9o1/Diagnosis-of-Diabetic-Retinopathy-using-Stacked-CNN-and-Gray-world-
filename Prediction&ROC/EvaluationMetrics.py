
""" 
ROC , Sensitvity , Specificty , and Classification Report analysis
"""

y_val_cat_prob=model.predict_proba(X_test)

from sklearn.metrics import roc_curve,roc_auc_score
 
fpr , tpr , thresholds = roc_curve ( y_test , y_val_cat_prob)

def plot_roc_curve(fpr,tpr): 
  plt.plot(fpr,tpr, color='red', label='ROC curve (AUC={:.3f})' % auc_score) 
  plt.axis([0,1,0,1]) 
  plt.xlabel('False Positive Rate') 
  plt.ylabel('True Positive Rate') 
  plt.show()

plot_roc_curve(fpr, tpr)



auc_score=roc_auc_score(y_test,y_val_cat_prob)

print(auc_score)

####################################################################################

from sklearn.metrics import confusion_matrix 
import sklearn.metrics as metrics

y_pred=model.predict(X_test )
print(y_pred[0:10])
print(y_test[0:10])



ypred_class = model.predict_classes(X_test, verbose=0)


print(ypred_class[0:10])

matrix = confusion_matrix(y_test, ypred_class)


print(matrix)                                       #PRINTING Confusion Matrix 

 
 
 """
 
 Below are the hardcoded values to calculate specific results such as sensitvity , specifcity etc  
 
 """
 
sensitivity = 165 / float(165 + 8)                #hardcoded Values can be changed, these were done to get the appropriate results.

print(sensitivity)   #Sensitivity: When the actual value is positive, how often is the prediction correct?

specificity = 157/ (157 + 8)        

print(specificity)   ##Specificity: When the actual value is negative, how often is the prediction correct?



false_positive_rate = 7 / float(7 + 199)

print(false_positive_rate)
print(1 - specificity)

precision = 244 / float(244 + 7)  

print(precision)


precision = 45 / (45 + 244)

print(precision)


"""
Printing Classification Report 

"""

from sklearn.metrics import classification_report

print(classification_report(y_test, ypred_class,target_names=lb.classes_))          #for generating the classwise report of the results 

