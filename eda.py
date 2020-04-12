# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 23:47:16 2019

@author: Dipto
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Loading the Raw Dataset
raw_df = pd.read_excel('Z-Alizadeh sani dataset.xlsx')
    
#Dropping the required feature
raw_df = raw_df.drop(columns=['BBB'])
    
#Statistical Analysis
def stats():
    inf = raw_df.info(verbose=False)    
    print(inf)
    print('Missing Values:',np.any(raw_df.isnull()) == True)
            
"""
  Visualisation of Some of the
  Features in the Dataset 
"""
    
#Percent of Male and Female patients
def maleAndFemale():
    male = len(raw_df[raw_df['Sex'] == 'Male'])
    female = len(raw_df[raw_df['Sex'] == 'Fmale'])
            
    plt.figure(figsize=(8,6))
            
    # Data to plot
    labels = 'Male','Female'
    sizes = [male,female]
    colors = ['#a4d7e1','#d1eecc']
    explode = (0, 0)  # explode 1st slice
             
    # Plot
    plt.title(label='Distribution of Male and Female patients', loc='center')
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,
    autopct='%1.1f%%', shadow=True, startangle=90)
             
    plt.axis('equal')
    plt.show()
        
#Percentage of CAD in Dataset
def percentOfCad():
    male = len(raw_df[raw_df['Cath'] == 'Normal'])
    female = len(raw_df[raw_df['Cath']== 'Cad'])
           
    plt.figure(figsize=(8,6))
            
    # Data to plot
    labels = 'Normal','CAD'
    sizes = [male,female]
    colors = ['#a4d7e1','#d1eecc']
    explode = (0, 0)  # explode 1st slice
    
    # Plot
    plt.title(label='Distribution of CAD and Normal patients')
    plt.pie(sizes, explode=explode, labels=labels, colors=colors,autopct='%1.1f%%',
    shadow=True, startangle=90)
    plt.axis('equal')
    plt.show()    
                
#CAD Patients by Age    
def cadPatientsByAge():
    plt.figure(figsize=(15,6))
    plt.title(label='Distribution of CAD and Normal patients by Age',loc='center')
    sns.countplot(x='Age',data = raw_df, hue ='Cath', palette='Blues')
    plt.ylabel("Number of Patients")
    plt.show()
                  
#CAD by Gender
def cadPatientsBySex():
    plt.figure(figsize=(15,6))
    plt.title(label='Distribution of CAD and Normal patients by Sex',loc='center')
    sns.countplot(x='Sex',data = raw_df, hue ='Cath', palette='Blues')
    plt.ylabel("Number of Patients")
    plt.show()
        
#CAD by FH
def cadPatientsByFH():
    plt.figure(figsize=(15,6))
    plt.title(label='Distribution of CAD and Normal patients by Family History',loc='center')
    sns.countplot(x='FH',data = raw_df, hue ='Cath', palette='Blues')
    plt.ylabel("Number of Patients")
    plt.xlabel('0 for Yes and 1 for No')
    plt.show()  
                            
#CAD by Current Smoker
def cadbyCS():
    plt.figure(figsize=(15,6))
    plt.title(label='Distribution of CAD and Normal patients by Current Smoker')
    sns.countplot(x='Current Smoker',data=raw_df,palette='Blues',hue='Cath')
    plt.xlabel('0 for Yes and 1 for No')
    plt.ylabel("Number of Patients")
    plt.show()
    
#CAD by EX-Smoker
def cadbyES():
    plt.figure(figsize=(15,6))
    plt.title(label='Distribution of CAD and Normal patients by EX-Smoker')
    sns.countplot(x='EX-Smoker',data=raw_df,palette='Blues',hue='Cath')
    plt.xlabel('0 for Yes and 1 for No')
    plt.ylabel("Number of Patients")
    plt.show()
    
#CAD by Obesity
def cadbyObesity():
    plt.figure(figsize=(15,6))
    plt.title(label='Distribution of CAD and Normal patients by Obesity',loc='center')
    sns.countplot(x='Obesity',data = raw_df, hue ='Cath', palette='Blues')
    plt.ylabel("Number of Patients")
    plt.show()
        
#CAD by Typical Chest Pain
def cadbyTCP():
    plt.figure(figsize=(15,6))
    plt.title(label='Distribution of CAD and Normal patients by Typical Chest Pain')
    sns.countplot(x='Typical Chest Pain',data=raw_df,palette='Blues',hue='Cath')
    plt.xlabel('0 for Yes and 1 for No')
    plt.ylabel("Number of Patients")
    plt.show()
    
#CAD by VHD
def cadbyVHD():
    plt.figure(figsize=(15,6))
    plt.title(label='Distribution of CAD and Normal patients by Valvular Heart Disease',loc='center')
    sns.countplot(x='VHD',data = raw_df, hue ='Cath', palette='Blues')
    plt.ylabel("Number of Patients")
    plt.show()
        
#CAD by HTN
def cadPatientsByHTN():
    plt.figure(figsize=(15,6))
    plt.title(label='Distribution of CAD and Normal patients by Hypertension',loc='center')
    sns.countplot(x='HTN',data = raw_df, hue ='Cath', palette='Blues')
    plt.ylabel("Number of Patients")
    plt.xlabel('0 for Yes and 1 for No')
    plt.show()  
    
#CAD by Blood Pressure
def cadPatientsByBP():
    plt.figure(figsize=(15,6))
    plt.title(label='Distribution of CAD and Normal patients by Blood Pressure',loc='center')
    sns.countplot(x='BP',data = raw_df, hue ='Cath', palette='Blues')
    plt.ylabel("Number of Patients")
    plt.show()
    
#CAD by Diabetes Mellitus
def cadPatientsByDM():
    plt.figure(figsize=(15,6))
    plt.title(label='Distribution of CAD and Normal patients by Diabetes Mellitus',loc='center')
    sns.countplot(x='DM',data = raw_df, hue ='Cath', palette='Blues')
    plt.ylabel("Number of Patients")
    plt.xlabel('0 for Yes and 1 for No')
    plt.show()
    
    
