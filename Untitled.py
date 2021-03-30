#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 


# In[2]:


import numpy as np


# In[3]:


train_data = pd.read_csv("train.csv")


# In[4]:


train_data.head()


# In[5]:


test_data = pd.read_csv("test.csv")


# In[6]:


test_data.head()


# In[7]:


X_train = train_data.iloc[:,:-1]
X_train.tail()


# In[8]:


y_train = train_data["SalePrice"]
y_train.head()


# In[9]:


X = train_data.iloc[:,:-1]


# In[10]:


X["Alley"].fillna("NoAlley",inplace = True)
X["PoolQC"].fillna("NoPool",inplace = True)
X["Fence"].fillna("NoFence",inplace = True)
X["MiscFeature"].fillna("NoMisc",inplace = True)

X.tail()


# In[11]:


x = train_data.iloc[:,:-1]


# In[12]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy ='most_frequent')
#features = ["Alley","PoolQC","Fence","MiscFeature"]
imputer.fit(x)
x = imputer.transform(x)
x


# In[13]:


y_train.to_numpy()


# In[14]:


new_x = test_data.iloc[:,:-1]
new_x


# In[15]:


new_y = np.array(test_data.iloc[:,-1])
new_y


# In[16]:


new_y.size


# In[17]:


new_x.size


# In[18]:


new_y.size


# In[19]:


test_data.size


# In[20]:


new_x = test_data.iloc[:,:-1]


# In[21]:


new_x.size


# In[22]:


new_x


# In[23]:


new_x.shape


# In[24]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputer.fit(new_x)
new_x= imputer.transform(new_x)
new_x


# In[25]:


from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(new_x)

enc.transform(new_x).toarray()
new_x


# In[26]:


new_x = pd.DataFrame(new_x)


# In[27]:


new_x


# In[29]:


from sklearn.preprocessing import LabelEncoder
temp =new_x.apply(LabelEncoder().fit_transform)



# In[30]:


temp


# In[31]:


temp.to_numpy()


# In[32]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
enc.fit(new_y)
new_y = enc.transform(new_y)
new_y


# In[33]:


from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(temp, new_y)


# In[34]:


from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(temp, new_y)


# In[35]:



y_pred = classifier.predict(temp)
y_pred


# In[36]:


from sklearn.metrics import accuracy_score
accuracy_score(new_y,y_pred)


# In[37]:


x_train = train_data.iloc[:,:-1].values
y_train = train_data.iloc[:,-1].values
x_train


# In[38]:


x_train.size


# In[39]:


y_train.size


# In[40]:


from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
enc.fit(y_train)
y_train = enc.transform(y_train)


# In[41]:


y_train


# In[46]:


x_train = pd.DataFrame(x_train)

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
imputer.fit(x_train)
new_x= imputer.transform(x_train)
x_train
# In[43]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(x_train)
x_train = imputer.transform(x_train)


# In[44]:


x_train


# In[47]:


from sklearn.preprocessing import LabelEncoder
the_x =x_train.apply(LabelEncoder().fit_transform)


# In[48]:


the_x = 


# In[49]:


the_x


# In[50]:


the_x.to_numpy()


# In[51]:


y_train


# In[52]:


from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 100,criterion ='entropy' , random_state=0)
classifier.fit(the_x,y_train)


# In[53]:


y_pred = classifier.predict(the_x)


# In[54]:


y_pred


# In[55]:


from sklearn.metrics import accuracy_score
accuracy_score(y_train,y_pred)


# In[56]:


x_test = test_data


# In[57]:


x_test


# In[58]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(x_test)
x_test = imputer.transform(x_test)


# In[59]:


x_test = pd.DataFrame(x_test)


# In[60]:


from sklearn.preprocessing import LabelEncoder
test_x =x_test.apply(LabelEncoder().fit_transform)


# In[61]:


test_x


# In[62]:


test_x.to_numpy()


# In[64]:


final_X = train_data.iloc[:,:-1]
final_y = train_data.iloc[:,-1]


# In[65]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(final_X)
final_X = imputer.transform(final_X)


# In[66]:


final_X = pd.DataFrame(final_X)


# In[68]:


from sklearn.preprocessing import LabelEncoder
final_x =final_X.apply(LabelEncoder().fit_transform)


# In[70]:


final_x.to_numpy()


# In[72]:


final_y.to_numpy()


# In[73]:


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=10,random_state=0)
regressor.fit(final_x,final_y)


# In[74]:


final_test = test_data


# In[76]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
imputer.fit(final_test)
final_test = imputer.transform(final_test)


# In[77]:


final_test  = pd.DataFrame(final_test)


# In[78]:


from sklearn.preprocessing import LabelEncoder
final_Test =final_test.apply(LabelEncoder().fit_transform)


# In[79]:


final_Test


# In[80]:


y_pred = regressor.predict(final_Test)


# In[82]:


y_pred = pd.DataFrame(y_pred,columns=['SalePrice'])


# In[83]:


y_pred


# In[84]:


ids = test_data["Id"]


# In[85]:


ids


# In[87]:


ids = pd.DataFrame(ids,columns = ['Id'])


# In[88]:


ids


# In[89]:


result = ids.join(y_pred)


# In[90]:


result


# In[100]:


result.to_csv('result1.csv',index=False)

