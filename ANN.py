# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 19:19:37 2019

@author: Dipto
"""

from pre_process import *
from numpy import interp
from pre_process import *
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold(n_splits=10,random_state=42)
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(55,23),random_state=42,max_iter=500)
mlp_bal = MLPClassifier(hidden_layer_sizes=(55,23),random_state=42,max_iter=500) 

class ANN:
     def __init__(self,X_train,X_train_res,X_test,y_train,y_train_res,y_test):
        self.X_train = X_train
        self.X_train_res = X_train_res
        self.X_test = X_test
        self.y_train = y_train
        self.y_train_res = y_train_res
        self.y_test = y_test
        
     def ann_imbal(self):
        mlp.fit(self.X_train,self.y_train)
        mlp_pred = mlp.predict(self.X_test)
        
        #Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm=confusion_matrix(self.y_test,mlp_pred)
        conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
        plt.figure(figsize = (8,5))
        plt.title(label='Confusion Matrix of Artificial Neural Network trained on the imbalanced dataset')
        sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="Blues")
        plt.show()
        
        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(mlp,self.X_train,self.y_train,cv=10)
        print('%.2f'%(accuracies.mean()*100),'%')
        print('%0.2f'%(accuracies.std()*100))
                        
     def roc_auc_ann_imbal(self):        
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        plt.figure(figsize=(10,10))
        i = 0
        for train, test in cv.split(self.X_train,self.y_train):
            probas_ = mlp.fit(self.X_train[train],self.y_train[train]).predict_proba(self.X_train[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(self.y_train[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
        
            i += 1
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)
        
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
        
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate',fontsize=18)
        plt.ylabel('True Positive Rate',fontsize=18)
        plt.title('ROC-AUC of Artificial Neural Network trained on the imbalanced dataset',fontsize=18)
        plt.legend(loc="lower right", prop={'size': 15})
        plt.grid()
        plt.show()
                
     def ann_bal(self):
        mlp_bal.fit(self.X_train_res,self.y_train_res)
        mlp_pred_bal = mlp.predict(self.X_test)
        
        #Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm=confusion_matrix(self.y_test,mlp_pred_bal)
        conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
        plt.figure(figsize = (8,5))
        plt.title(label='Confusion Matrix of Artificial Neural Network trained on the balanced dataset')
        sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="Blues")
        plt.show()
        
        # Applying k-Fold Cross Validation
        accuracies = cross_val_score(mlp_bal,self.X_train_res,self.y_train_res,cv=10)
        print('%.2f'%(accuracies.mean()*100),'%')
        print('%0.2f'%(accuracies.std()*100))
                                
     def roc_auc_ann_bal(self):        
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        plt.figure(figsize=(10,10))
        i = 0
        for train, test in cv.split(self.X_train_res,self.y_train_res):
            probas_ = mlp_bal.fit(self.X_train_res[train],self.y_train_res[train]).predict_proba(self.X_train_res[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(self.y_train_res[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
        
            i += 1
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Chance', alpha=.8)
        
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)
        
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
        
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate',fontsize=18)
        plt.ylabel('True Positive Rate',fontsize=18)
        plt.title('ROC-AUC of Artificial Neural Network trained on the balanced dataset',fontsize=18)
        plt.legend(loc="lower right", prop={'size': 15})
        plt.grid()
        plt.show()
        
       
       
        
        
        



        
    