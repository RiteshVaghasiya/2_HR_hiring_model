#!/usr/bin/env python
# coding: utf-8

# # We build a machine learning model for HR department that can help them decide salaries for future candidates based on experience of candidate, his written test score and personal interview score.

# In[ ]:


import pandas as pd
import numpy as np
from sklearn import linear_model
from word2number import w2n


# In[29]:


d = pd.read_csv("hiring.csv")
d


# # Data Preprocessing: Fill zero values inplace of NaN experience column

# In[30]:


d.experience = d.experience.fillna("zero")
d


# In[31]:


d.experience = d.experience.apply(w2n.word_to_num)
d


# # Data Preprocessing: Fill NA values with median value of a test_score column

# In[34]:


import math
median_test_score = math.floor(d['test_score(out of 10)'].mean())
median_test_score


# In[35]:


d['test_score(out of 10)'] = d['test_score(out of 10)'].fillna(median_test_score)
d


# In[36]:


reg = linear_model.LinearRegression()
reg.fit(d[['experience','test_score(out of 10)','interview_score(out of 10)']],d['salary($)'])


# Find new candidate salary of 2 years experience, 9 test_score(out of 10) & 6 interview_score(out of 10)

# In[38]:


reg.predict([[2,9,6]])


# In[ ]:


Find new candidate salary of 12 years experience, 10 test_score(out of 10) & 10 interview_score(out of 10)


# In[39]:


reg.predict([[12,10,10]])

