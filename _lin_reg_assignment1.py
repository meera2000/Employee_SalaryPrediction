#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error


# In[3]:


cd D:\FT\python\Ml\Assignment


# Part-1: data Exploration and Pre-processing

# In[ ]:


# 1) Load the given dataset 


# In[28]:


data=pd.read_csv("Python_Linear_Regres_a1.csv")


# In[29]:


data.head()


# In[30]:


# 2) Fill Null value of experience column with the value
# data['experience'].isnull().sum()
data['experience'].fillna('Zero',inplace=True)


# In[32]:


math.floor(data['test_score(out of 10)'].mean())


# In[33]:


# 3) Replace the Null values of the column test score with mean value
import math
data['test_score(out of 10)'].fillna(math.floor(data['test_score(out of 10)'].mean()),inplace=True)


# In[34]:


data.isnull().sum()


# In[35]:


# 4) Display a scatter plot between experience and Salary 
x=data['experience']
y=data['salary($)']
plt.scatter(x,y)
plt.show()


# In[36]:


# 5) Display a scatter plot between test score and Salary 
x=data['test_score(out of 10)']
y=data['salary($)']
plt.scatter(x,y)
plt.show()


# In[37]:


# 6) Display a scatter plot between interview score and Salary 
x=data['interview_score(out of 10)']
y=data['salary($)']
plt.scatter(x,y)
plt.show()


# In[38]:


# 7) Display bar plot for experience
x=data['experience']
y=data['salary($)']
plt.bar(x,y)
plt.show()


# Part-2: Working with Model

# In[39]:


# 1) Separate feature data from target data
x=data.drop('salary($)',axis=1)
y=data[['salary($)']]
x


# In[40]:


x.dtypes
x.shape


# In[41]:


data


# In[26]:





# In[18]:


y.dtypes
y.shape


# In[43]:


get_ipython().system('pip install word2number')


# In[42]:


from word2number import w2n
list1=[]
for i  in df['experience']:
    list1.append(w2n.word_to_num(i))
    
df['new_experience']=list1
df


# In[19]:


# 2) Create a Linear regression model between Features and target data
model=LinearRegression()


# In[20]:


model.fit(x,y)


# In[21]:


predicted_salary=model.predict(x)


# In[ ]:


# 3) Display the test score and training score
model.score(y,predicted_salary)


# In[ ]:


# 4) Extract slope and intercept value from the model
print('Slope:' ,model.coef_) 
print('Intercept:',model.intercept_) 


# In[53]:


# 5) Display Mean Squared Error
# 6) Display Mean Absolute Erro
# 7) Display Root mean Squared error
# 8) Display R2 score
print("Mean-squared-error",mean_squared_error(y,predicted_salary))
print("Mean Absolute Error",mean_absolute_error(y,predicted_salary))
print("Root mean Squared error",math.root(mean_squared_error(y,predicted_salary)))
print("R2 score=",r2_score(y,predicted_salary))

