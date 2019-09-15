# coding: utf-8

# In[2]:


#Linear Regression model program
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn import linear_model
import matplotlib.pyplot as plt

# In[3]:


mydata = [[151,63],[174,81],[138,56],[186,91],[128,47],[136,57],[179,76],[163,72],[152,62],[131,48]]


# In[4]:


df1 = pd.DataFrame(mydata,columns=['ht','wt'])
print(df1)


# In[6]:


X = df1.iloc[:,:-1]
y = df1.iloc[:,1]
print(X,y)

# In[7]:


#import sklearn train-test-split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.5, random_state=0)


# In[8]:


print('X Train')


# In[9]:


print(X_train)
print('x test')
print(X_test)


# In[10]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)
print("Coef ::")
print(regressor.coef_)
print("intercept ::")
print(regressor.intercept_)


# In[11]:


#predict weight of new height
new = [[180]] #requires 2D array for predict function
print('predicated value of 180 cms ')
print(regressor.predict(new))


# In[12]:


y_pred = regressor.predict(X_test)
print('accuracy of LR model')
print(regressor.score(X_test,y_test))


# In[13]:


plt.xlabel('Height')
plt.ylabel('weight')


# In[14]:


plt.scatter(X_train,y_train,color='red')
plt.scatter(X_test,y_test,color='black',marker = '*')
plt.plot(X_test,y_pred,color = 'grey')
plt.scatter(new,regressor.predict(new), color = 'green')
plt.show()
