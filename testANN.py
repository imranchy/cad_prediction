# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 17:15:03 2019

@author: Dipto
"""

from pre_process import *
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(66,45,))
mlp.fit(X_train_res,y_train_res.ravel())
mlp_pred = mlp.predict(X_test)



from sklearn.metrics import confusion_matrix,classification_report
print('Before PCA',classification_report(y_test,mlp_pred))
print(confusion_matrix(y_test,mlp_pred),'\n')

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = mlp, X = X_train_res, y = y_train_res.ravel(), cv = 10)
print('Accuracies before PCA:',(accuracies.mean()*100),'%')
print('Standard Deviation:','+/-',(accuracies.std()*100),'%')

mlp_pca = MLPClassifier(hidden_layer_sizes=(35,24,))
mlp.fit(X_train_pca,y_train_res.ravel())
mlp_pred_pca = mlp.predict(X_test_pca)


print('After PCA',classification_report(y_test,mlp_pred_pca))
print(confusion_matrix(y_test,mlp_pred_pca))

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = mlp_pca, X = X_train_pca, y = y_train_res.ravel(), cv = 10)
print('Accuracies After PCA:',(accuracies.mean()*100),'%')
print('Standard Deviation:','+/-',(accuracies.std()*100),'%')
