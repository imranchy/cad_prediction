# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 15:52:57 2019

@author: Dipto
"""

from LogReg import *
from SVM import *
from ANN import *

if __name__ == "__main__":
    
    n = input('Enter choice of operation: ')
                     
    if n=='EDA':
        #Statistical Analysis
        stats()
        #Distribution of Important features of CAD
        maleAndFemale()
        percentOfCad()
        cadPatientsByAge()
        cadPatientsBySex()
        cadPatientsByFH()
        cadbyCS()
        cadbyES()
        cadbyObesity()
        cadbyTCP()
        cadbyVHD()
        cadPatientsByHTN()
        cadPatientsByBP()
        cadPatientsByDM()

    elif n == 'Prepare':
        before_res()
        after_res()
        cadbyecp()
        
    elif n == 'LR':
        lr = LR(X_train,X_train_res,X_test,y_train,y_train_res,y_test)
        lr.logreg_imbal()
        lr.logreg_bal()
        lr.roc_auc_logreg_imbal()
        lr.roc_auc_logreg_bal()    
        
    elif n == 'SVM':
        svm = SVM(X_train,X_train_res,X_test,y_train,y_train_res,y_test)
        svm.svm_imbal()
        svm.roc_auc_svm_imbal()
        svm.svm_bal()
        svm.roc_auc_svm_bal()
                                                                        
    elif n == 'ANN':
        ann = ANN(X_train,X_train_res,X_test,y_train,y_train_res,y_test)
        ann.ann_imbal()
        ann.roc_auc_ann_imbal()
        ann.ann_bal()
        ann.roc_auc_ann_bal()
                        

        
        
        
        
        
        
 
    
    
    
    
    
    
    
    
    
    
    
    
    

    
