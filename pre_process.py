# -*- coding: utf-8 -*-
"""
Created on Sun Aug 11 00:17:59 2019

@author: Dipto
"""

from eda import *

# Processed dataset to be used for Machine Learning Models and also avoiding the Dummy Variable Trap 
processed_df = pd.get_dummies(raw_df,drop_first=True)
    
#CAD by Exertional CP
def cadbyecp():
    plt.figure(figsize=(15,6))
    plt.title(label='Distribution of CAD and Normal patients by Exertional Chest Pain',loc='center')
    sns.countplot(x='Exertional CP',data = raw_df, hue ='Cath', palette='Blues')
    plt.ylabel("Number of Patients")
    plt.show()
    
#Taking the X and y vectors from the Processed Dataset
X = processed_df.iloc[:,:-1].values
y = processed_df.iloc[:,55].values

#Splitting the Data for applying Machine Learning Algorithms
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        
#Feature Scaling of the Training and Test Sets
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
    
#Importing SMOTE        
from imblearn.over_sampling import SMOTE
    
# Visualisation of class distributions before and after SMOTE 
def before_res():
#Checking for CAD Distribution
    plt.figure(figsize=(7,5))
    sns.countplot(processed_df['Cath_Normal'],palette='Blues')
    plt.title("Count of Target Class")
    plt.xlabel("0 for CAD and 1 for Normal")
    plt.ylabel("Count of Target Class")
    plt.show()
        
#Applicqation of SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())
        
#After SMOTE
def after_res():
    plt.figure(figsize=(10,5))
    sns.countplot(y_train_res,palette='Blues')
    plt.title("Count of Target Class")
    plt.xlabel("0 for CAD and 1 for Normal")
    plt.ylabel("Count of Target Class")
    plt.show()
    

    





















    
    
    
    



