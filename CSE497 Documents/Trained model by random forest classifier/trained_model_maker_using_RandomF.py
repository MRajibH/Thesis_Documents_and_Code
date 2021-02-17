#!/usr/bin/env python
# coding: utf-8

# In[132]:


import joblib
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


# In[133]:


data = pd.read_csv(
    r"C:\Users\lenovo\Desktop\Thesis\Dataset final\suicide_dataset_numeric.csv")


# In[134]:


# In[135]:


# In[136]:


features = data.columns
features


# In[137]:


features = [x for x in features if x != 'attempt_suicide' and x != 'Age']
features


# In[138]:


train, test = train_test_split(data, test_size=0.3)
print(len(data))
print(len(train))
print(len(test))


# In[139]:


x_train = train[features]
y_train = train["attempt_suicide"]

x_test = test[features]
y_test = test["attempt_suicide"]


# In[140]:


Rf = RandomForestClassifier(n_estimators=100)


# In[141]:


Rf = Rf.fit(x_train, y_train)


# In[142]:


y_pred = Rf.predict(x_test)


# In[143]:


y_pred


# In[144]:


score = accuracy_score(y_test, y_pred) * 100
print("Accuracy using Random forest: ", round(score, 1), "%")


# In[145]:


# In[146]:


print(confusion_matrix(y_test, y_pred))


# In[ ]:

# In[ ]:
filename = 'finalized_modeltest.sav'
joblib.dump(Rf, filename)
