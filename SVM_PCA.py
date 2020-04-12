# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 01:23:59 2019

@author: Dipto
"""

from pre_process import *

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.fit_transform(X_test)

class SVM_PCA: 
    def __init__(self,X_train_pca,X_test_pca,y_train,y_test):
        self.X_train_pca = X_train_pca
        self.X_test_pca = X_test_pca
        self.y_train = y_train
        self.y_test = y_test
        

    def applySVM(self):
        #Fitting SVM to the Training set
        from sklearn.svm import SVC
        classifier = SVC(kernel='rbf')
        classifier.fit(self.X_train_pca, self.y_train)
        
        
        #Predicting the Test set results of SVM
        SVM_y_pred = classifier.predict(self.X_test_pca)
        
        
        #Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm=confusion_matrix(self.y_test,SVM_y_pred)
        conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
        plt.figure(figsize = (8,5))
        plt.title(label='Confusion Matrix of Support Vector Machine')
        sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="YlGnBu")
        plt.show()
        
pca_svm = SVM_PCA(X_train_pca,X_test_pca,y_train,y_test)
pca_svm.applySVM()
