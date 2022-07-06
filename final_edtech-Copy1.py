#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[2]:





# In[3]:


# loading data
edtech = pd.read_csv(r'C:\Users\sahil\Documents\360 DigitMG\Project\project attempt2\edtech_price_prediction.csv')


# In[4]:


# shape of data
edtech.shape


# In[5]:


# Finding datatypes
edtech.info()


# In[6]:


# dropping column
edtech = edtech.iloc[:,2:]


# In[7]:


# checking for duplicates values

edtech.duplicated().sum()


# In[8]:


# dropping the duplicate values
edtech.drop_duplicates(inplace=True)


# In[9]:


edtech.shape


# In[10]:


edtech.columns


# In[11]:


# checking for missing values
edtech.isnull().sum()


# In[12]:


# checking for zero variance

edtech.var()==0


# In[13]:


edtech.describe()


# In[14]:


edtech.columns


# In[15]:


# Exploratory Data Analysis

edtech.mean()


# In[16]:


# Median
edtech.median()


# In[17]:


# 2nd Business moment
# Variance
edtech.var()


# In[18]:


#Standard Deviation

edtech.std()


# In[19]:


# Skewness

edtech.skew()


# In[20]:


# Kurtosis

edtech.kurt()


# In[21]:


edtech.columns


# In[22]:


sns.countplot(edtech.institute_brand_value)


# In[23]:


sns.countplot(edtech.course_title)


# In[24]:


sns.countplot(edtech.course_market)


# In[25]:


sns.countplot(edtech.online_live_class)


# In[26]:


sns.countplot(edtech.online_pre_recorded_sessions)


# In[27]:


sns.countplot(edtech.Offiline_classes)


# In[28]:


sns.countplot(edtech.state)


# In[29]:


sns.countplot(edtech.instructors_grade)


# In[30]:


sns.countplot(edtech.course_Level)


# In[31]:


sns.countplot(edtech.Infrastructure_cost)


# In[32]:


sns.countplot(edtech.cost_of_course_curricullum)


# In[33]:


sns.countplot(edtech.competition_level)


# In[34]:


sns.countplot(edtech.certification)


# In[35]:


sns.countplot(edtech.Placement)


# In[36]:


sns.distplot(edtech.study_material_cost)


# In[37]:


sns.distplot(edtech.number_of_instructors)


# In[38]:


sns.distplot(edtech.Office_rent)


# In[39]:


sns.distplot(edtech.Office_electricity_charges)


# In[40]:


sns.distplot(edtech.Misclleneous_expense)


# In[41]:


sns.distplot(edtech.cost_of_acquisition)


# In[42]:


sns.distplot(edtech.price)


# In[43]:


# checking for outliers

sns.boxplot(edtech.number_of_instructors)


# In[44]:


sns.boxplot(edtech.course_length)


# In[45]:


sns.boxplot(edtech.Office_rent)


# In[46]:


edtech.columns


# In[47]:


sns.boxplot(edtech.Office_electricity_charges)


# In[48]:


sns.boxplot(edtech.Misclleneous_expense)


# In[49]:


sns.boxplot(edtech.cost_of_acquisition)


# In[50]:


sns.boxplot(edtech.price)





# In[52]:


#removing outliers in course_length

from feature_engine.outliers import Winsorizer


# In[53]:


winsor = Winsorizer(capping_method='iqr', tail='both', fold=1.5, variables=['course_length'])


# In[54]:


edtech.course_length = winsor.fit_transform(edtech[['course_length']])


# In[55]:


sns.boxplot(edtech.course_length)


# In[56]:


# creating dummy variables for categorical data
edtech = pd.get_dummies(edtech, columns=['course_title','course_market','online_live_class', 'online_pre_recorded_sessions', 'Offiline_classes','state','support_Staff_required','certification','Placement'],drop_first=True)


# In[57]:


brand_value ={'High':1,'Low':0}
instruct_grade = {'A':1,'B':0}
cor_level = {'Advance':2,'Intermediate':1,'Beginners':0}
cur_cost={'High':2,'Medium':1,'Low':0}
infra_cost={'High':1,'Low':0}
comp_level ={'High':2,'Medium':1,'Low':0}


# In[58]:


edtech.columns


# In[59]:


# Assigning numerical values
edtech['institute_brand_value'] = edtech.institute_brand_value.map(brand_value)
edtech['instructors_grade']=edtech.instructors_grade.map(instruct_grade)
edtech['course_Level']=edtech.course_Level.map(cor_level)
edtech['cost_of_course_curricullum']=edtech.cost_of_course_curricullum.map(cur_cost)
edtech['Infrastructure_cost']=edtech.Infrastructure_cost.map(infra_cost)
edtech['competition_level']=edtech.competition_level.map(comp_level)


# In[60]:


# To show all the columns at once
pd.set_option('display.max_columns', None)
edtech


# In[61]:


# seperating target variable and predictors
x=edtech.drop(columns=['price'], axis=1)


# In[62]:


y=edtech.price


# In[63]:


# Normalization

def norn_func(i):
    x= (i- i.min())/(i.max()-i.min())
    return(x)


# In[64]:


# Applying normalization so that data can be on same scale
x_norm= norn_func(x)


# In[65]:


df= pd.concat([x_norm,y], axis=1)


# In[66]:


df.head()


# In[67]:



from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV, StratifiedKFold


# In[71]:


# spliting the data into train and test data
x_train, x_test, y_train, y_test = train_test_split(x_norm, y)


# In[72]:


gbr = GradientBoostingRegressor()


# In[73]:


# Defining parameters to be used in GridSearchCV
parameters ={'learning_rate': [0.1,0.2,0.3,0.4,0.5],
            'n_estimators' :[100,300,500,1000],
            'max_depth': [2,3,5]}


# In[74]:


grid_gbr = GridSearchCV(estimator=gbr, param_grid=parameters, cv=5, n_jobs=-1)


# In[75]:



grid_gbr.fit(x_train,y_train)


# In[76]:


print("\n The best estimator across ALL searched params:\n",grid_gbr.best_estimator_)
print("\n The best score across ALL searched params:\n",grid_gbr.best_score_)
print("\n The best parameters across ALL searched params:\n",grid_gbr.best_params_)


# In[77]:


# Initializing gradient Boosting algorithm with best estimators obtained from GridSearchCV
best_gbr = grid_gbr.best_estimator_


# In[78]:


# prediction on test data
test_pred = best_gbr.predict(x_test)


# In[79]:


# importing evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score


# In[80]:


print("RMSE on Test data",np.sqrt(mean_squared_error(y_test,test_pred)))
print("R-Squared on Test data ", r2_score(y_test, test_pred))


# In[81]:


# predicting on train data with best gradient Boosting Algorithm model obtained from GridSearchCV
train_pred=best_gbr.predict(x_train)


# In[82]:


# predicting model on train data
print("RMSE on Train data",np.sqrt(mean_squared_error(y_train,train_pred)))
print("R-Squared on Train data ", r2_score(y_train, train_pred))


# In[83]:


# Applying RandomizedSearchCV
stratifiedkf=StratifiedKFold(n_splits=3)
random_gb = RandomizedSearchCV(estimator = gbr, param_distributions= parameters ,n_iter = 10, cv= stratifiedkf, n_jobs=-1)


# In[84]:


# Fitting Gradient boosting resgression algorithm with RandomizedSearchCV
random_gb.fit(x_train,y_train)


# In[85]:


# parameters used for best model
random_gb.best_params_


# In[86]:


random_gb.best_estimator_


# In[87]:


# intializing model with best estimator
random_gbmodel= random_gb.best_estimator_


# In[88]:


# predicting on test data from best results obtained from randomSearchCV
test_pred_ran = random_gbmodel.predict(x_test)


# In[89]:


print(" RMSE for Test data with RandomSearchCV is ", np.sqrt(mean_squared_error(y_test, test_pred_ran)))
print(" R- Squared score for Test result with RandomsearchCV is ", r2_score(y_test, test_pred_ran))


# In[90]:


# predicting on train data with best results obtained from randomSearchCV
train_pred_ran = random_gbmodel.predict(x_train)


# In[91]:


print(" RMSE for Train data with RandomSearchCV is ", np.sqrt(mean_squared_error(y_train, train_pred_ran)))
print(" R- Squared score for Test result with RandomsearchCV is ", r2_score(y_train, train_pred_ran))


# In[92]:


random_gbmodel.fit(x_norm.values,y.values)


# In[93]:


import pickle


# In[94]:


# making a pickle file
pickle_out = open("edtech_flask.pkl","wb")
pickle.dump(random_gbmodel,pickle_out)





