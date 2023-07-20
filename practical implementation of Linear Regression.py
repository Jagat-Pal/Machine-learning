#!/usr/bin/env python
# coding: utf-8

# # import required library

# In[1]:


import numpy as np  # numpy used for mathematics analysis
import pandas as pd   # pandas used for data manipulation
import matplotlib.pyplot as plt   # matplotlib used for data visualization
import seaborn as sns  # seaborn used for data visualization


# In[5]:


df = pd.read_csv("placement (1).csv")


# # How big is the data

# In[6]:


df.shape


# In[7]:


# In this dataset 200 rows and 2 columns


# # How does the data looklike

# In[9]:


df.sample(6) # this method exactret row randomly from the dataset


# # find the information of the data

# In[10]:


df.info()


# # How does the data looklike mathematically

# In[12]:


df.describe()


# # Are there any missing value in the data

# In[13]:


df.isnull().sum()


# In[14]:


# threre is no missing value in dataset


# # Are there any duplicated value in the dataset

# In[15]:


df.duplicated().sum()


# In[16]:


# there is no duplicate value in the dataset


# # What is the correlation between column

# In[17]:


df.corr()


# # Distribution of the dataset

# In[29]:


plt.scatter(df["cgpa"], df["package"])
plt.xlabel("CGPA")
plt.ylabel("PACKAGE")


# # How import warnings

# In[19]:


import warnings
warnings.filterwarnings("ignore")


# # find the outlier in dataset in particular column

# In[21]:


sns.boxplot(df["cgpa"])
plt.show()


# In[22]:


# Above boxplot represent there is no outlier in cgpa column


# In[23]:


sns.boxplot(df["package"])


# In[24]:


# above boxplot represent there is no outlier in package column


# # Find independent and dependent feature

# In[25]:


x = df.iloc[:, 0] # independent feature
y = df.iloc[:, 1]  # dependent feature


# In[26]:


x


# In[27]:


y


# # Split dataset into train test

# In[28]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 42)

# x is independent feature, y is dependent feature, test_size is 0.25 means, test data is 25 percent
# training data is 75 percent, random_state used to fixed the suffling process
# random_state = 42 means, 42 value give better performance compare to other value (like 1, 2, 4,7, 9, 10, etc)


# # Reshape the dataset

# Reshaping is did because x, y is one dimensional

# In[55]:


x = np.array(x)
x = x.reshape(-1, 1)


# In[47]:


x_train = np.array(x_train)  # dataframe convert into array
x_train = x_train.reshape(-1,1) # reshape one dimension array


# In[48]:


x_test = np.array(x_test)   # dataframe convert into array
x_test = x_test.reshape(-1,1) # reshape one dimension array


# In[49]:


y_train = np.array(y_train)  # dataframe convert into array
y_test = np.array(y_test)
y_train = y_train.reshape(-1, 1) # reshape one dimension array
y_test = y_test.reshape(-1, 1)


# # Import the model and training the model

# In[50]:


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
lr = regression.fit(x_train, y_train)


# # weight or coefficient and intercept

# In[51]:


lr.coef_    # After training of the model, weight is obtain


# In[53]:


lr.intercept_   # After training of the model, intercept is obtain


# # Bestfit line or regression line

# In[56]:


plt.scatter(df["cgpa"], df["package"])
plt.plot(x, lr.predict(x), color = "red")
plt.xlabel("CGPA")
plt.ylabel("PACKAGE")
plt.show()


# # Prediction the model

# In[59]:


y_pred = lr.predict(x_test)
y_pred


# # Regression metrics

# In[57]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[66]:


MSE = mean_squared_error(y_test, y_pred)*100
MSE


# In[62]:


# 9.12 percent is mean squared error


# In[63]:


print("MAE", mean_absolute_error(y_test, y_pred)*100)


# In[65]:


print("R2 score", r2_score(y_test, y_pred)*100)


# In[69]:


import math
rmse = math.sqrt(MSE)
rmse


# # conclusion

# In our dataset there is two column, first is cgpa and second is package, cgpa is independent feature and package is dependent
# feature. mean squared error(MSE) and mean absolute error(MAE) represent the error of the model. MSE is less than MAE because
# in our dataset there is no outlier. The disadvantage of MSE is that error unit square so we find the
# root mean squared error(rmse) is 3.02 percent. R2 score 77.74 percent explain that 77.74 percent  cgpa is responsible in package
# variation.

# In[ ]:




