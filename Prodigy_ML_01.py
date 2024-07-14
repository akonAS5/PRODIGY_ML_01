#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


# Importing Dataset
house_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
# Summary  of our dataset
house_data.head()
house_data.describe()


# In[4]:


# Size of dataset
house_data.shape


# In[5]:


# Printing out all features 
features = house_data.columns
print("Features in the dataset:")
for feature in features:
    print(feature)


# In[6]:


house_data.info()    


# In[8]:


# Check for train missing values
missing_value= (house_data.isnull().sum())
print(missing_value[missing_value > 0])


# In[9]:


# Check for test missing values
missing_value = (test_data.isnull().sum())
print(missing_value[missing_value > 0]) 


# In[10]:


# Exploratory Data Analysis (EDA)
sns.histplot(house_data.SalePrice, bins=50)


# In[12]:


# Creating target and relevant feature variables
features = ['LotArea', 'YearBuilt','1stFlrSF', '2ndFlrSF', 
            'FullBath','HalfBath','BedroomAbvGr', 'TotRmsAbvGrd']
y = house_data.SalePrice
X = house_data[features]
X.head()


# In[13]:


# Data splitting into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Fitting
model = LinearRegression()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on testing data
y_pred = model.predict(X_test)
model.score(X_test, y_test)


# In[14]:


# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

 
# Display the evaluation metrics
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R-squared (R2): {r2:.2f}')


# In[15]:


# Plot actual vs. predicted values
plt.scatter(y_test, y_pred, c = "blue")
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs. P)redicted Prices (Linear Regression)')
plt.show()


# In[16]:


# Prediction of new houses on test data

test_data = pd.read_csv('test.csv')
test_data.head()

features = ['LotArea', 'YearBuilt','1stFlrSF', '2ndFlrSF', 
            'FullBath','HalfBath','BedroomAbvGr', 'TotRmsAbvGrd']

X = test_data[features]
prediction_new = model.predict(X)


# In[ ]:




