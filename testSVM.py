# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 20:52:53 2019

@author: Dipto
"""


from pre_process import *
from sklearn.metrics import *
from eda import *

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)
X_train_lda = lda.fit_transform(X_train_res,y_train_res)
X_test_lda = lda.transform(X_test)




from sklearn.linear_model import LogisticRegression

svc = LogisticRegression()


svc.fit(X_train_res, y_train_res)
        
#Predicting the Test set results of SVM
SVM_y_pred = svc.predict(X_test)
        
#Making the Confusion Matrix
cm=confusion_matrix(y_test,SVM_y_pred)
conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
plt.figure(figsize = (8,5))
plt.title(label='Confusion Matrix of Support Vector Machine of imbalanced dataset')
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="Blues")
plt.show()
        

from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(hidden_layer_sizes=(2,1,))
mlp.fit(X_train_lda,y_train_res)
mlp_pred = mlp.predict(X_test_lda)


print('Before PCA',classification_report(y_test,mlp_pred))
print(confusion_matrix(y_test,mlp_pred),'\n')


x1 = process_df['BMI']
y2 = process_df['Cath']


 #Making the Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm=confusion_matrix(self.y_test,mlp_pred)
        conf_matrix=pd.DataFrame(data=cm,columns=['Predicted:0','Predicted:1'],index=['Actual:0','Actual:1'])
        plt.figure(figsize = (8,5))
        plt.title(label='Confusion Matrix of Artificial Neural Network trained on the imbalanced dataset')
        sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="Blues")
        plt.show()
        
        from sklearn.metrics import classification_report
        print('Classification report of Artifial Neural Network trained on the imbalanced dataset',classification_report(self.y_test,mlp_pred))

dummies = pd.get_dummies(raw_df)

