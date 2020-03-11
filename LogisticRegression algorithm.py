#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing library to read dataset
import pandas as pd


# In[25]:


# loading dataset with pandas
filepath = "E:/LIFESTYLE/Programming/National AI/Kumasi Hive/Datasets/diabetes.csv"
df = pd.read_csv(filepath)


# In[26]:


# checking first five row of dataframe
df.head()


# In[27]:


# checking number of (rows, columns)
df.shape


# In[28]:


# checking entire information of dataframe
df.info()


# #### Understanding data 

# In[29]:


# statistical summary of dataframe(df)
df.describe()


# In[30]:


#Checking Class distribution
print(df.groupby('Outcome').size())


# In[31]:


df_features = df.drop('Outcome', axis=1)


# In[32]:


df_features.shape


# In[ ]:


# importing librabries for data visualization
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


# In[33]:


# box and whisker plots for each Input variable (univariate plots)
df_features.plot(kind='box', subplots=True,figsize=(9,9), layout=(3,3), sharex=False, sharey=False)
plt.show()


# In[34]:


# Histogram for each input variable (univariate plots)
df_features.hist(figsize=(9,9))
plt.show()


# In[37]:


# scatter matrix plot for each input variable (multivariate plots)
scatter_matrix(df_features, figsize=(15,15), range_padding=0.5)
plt.show()


# In[ ]:


# import library for preprocessing 
from sklearn.preprocessing import StandardScaler


# In[63]:


# Pre-processing data
df_array = df.values
X_array = df_array[:,0:8]
y_array = df_array[:,8]

# standardizing arrays
scaler = StandardScaler()
scaler.fit(X_array)
transform_X = scaler.transform(X_array)


# In[65]:


import numpy as np
np.set_printoptions(precision=3)
print(transform_X[0:5,:])


# In[47]:


# importing libraries for model building and evaluation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[66]:


# building model with unscaled data
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression(solver='liblinear')
results = cross_val_score(model, X_array, y_array, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[67]:


# building model with scaled data
kfold = KFold(n_splits=10, random_state=7)
model = LogisticRegression(solver='liblinear')

results = cross_val_score(model, transform_X, y_array, cv=kfold)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))


# In[68]:


# creating X(matrix) and y(vector) from original data
X = df_features.values
y = df['Outcome'].values


# In[79]:


# splitting data into training and validation sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)


# In[80]:


# building model with training sets and making predictions
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
predictions = model.predict(X_test)


# In[92]:


# Evaluating classification model
print('Accuracy_score: ', format(accuracy_score(y_test, predictions), '.3f'))
print('\nConfusion_matrix: \n', confusion_matrix(y_test, predictions))
print('\nClassification_report: \n',classification_report(y_test, predictions))


# In[ ]:




