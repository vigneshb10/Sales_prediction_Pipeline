#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_column', None)


# In[2]:


df=pd.read_excel('Sales Dataset.xlsx')
df.head()


# In[3]:


df['Opportunity Status'].value_counts()/len(df) *100


# In[4]:


df=df.rename(columns={'Technology\nPrimary':'Technology'})


# In[5]:


df=df.rename(columns={'B2B Sales Medium':'Business Sales Medium','Sales Stage Iterations':'Sales Stage','Opportunity Size (USD)':'Opportunity Size'})


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df=df.drop('Opportunity ID',axis=1)


# In[9]:


for column in df.columns:
    num_unique_values = df[column].nunique()
    print(f'Number of unique values in {column}: {num_unique_values}')


# In[10]:


for item in df.columns:
    if df[item].dtype == 'object':
        print(f'{item}')
        print(f'{df[item].value_counts()}')
        print('\n')
    else:
        pass


# In[11]:


cat_columns=[item for item in df.columns if df[item].dtype == object]
num_columns=[col for col in df.columns if df[col].dtype != object]


# In[12]:


cat_columns


# In[13]:


num_columns


# In[14]:


df.isnull().sum()


# In[15]:


df[df.duplicated()]


# In[16]:


def outlier_detection(df, columns):
    plt.figure(figsize=(20,6))
    sns.boxplot(data=df, x=columns)
    plt.title(f'Box plot for {columns}')


# In[17]:


for item in num_columns:
    outlier_detection(df, item)
    print(f'{df[item].describe()}')


# In[18]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

outliers = ((df< (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)


# In[19]:


outliers


# In[20]:


def outlier_threshold(data, col):
    iqr=df[col].quantile(0.75) - df[col].quantile(0.25)
    upper_threshold= df[col].quantile(0.75) + (1.5 * iqr)
    lower_threshold= df[col].quantile(0.25) - (1.5 * iqr)
    df[col]=df[col].clip(lower_threshold, upper_threshold)


# In[21]:


outlier_threshold(df, num_columns[0])


# In[22]:


for item in num_columns:
    outlier_detection(df, item)


# In[23]:


from scipy.stats import zscore
z_scores = zscore(df['Sales Velocity'])
abs_z_scores = np.abs(z_scores)
outliers = (abs_z_scores > 3).all(axis=0)


# In[24]:


outliers


# In[25]:


sns.histplot(df['Sales Velocity'], bins=15, kde=True)


# In[26]:


def count_plots(df, col):
    plt.figure(figsize=(15,5))
    sns.countplot(df[col])
    plt.title(f'Count plot for {col}')


# In[27]:


for item in cat_columns:
    count_plots(df, item)


# In[28]:


def count(df, col):
    plt.figure(figsize=(15,5))
    sns.countplot(df[col], hue=df['Opportunity Status'])
    plt.title(f'Count plot for {col}')


# In[29]:


for item in cat_columns:
    count(df, item)


# In[30]:


sns.pairplot(df, hue='Opportunity Status')


# In[31]:


df.head()


# In[32]:


cat_columns


# In[33]:


df1=df.copy()
df1=pd.get_dummies(df1[['Technology','City','Business Sales Medium','Client Revenue Sizing','Client Employee Sizing','Business from Client Last Year','Compete Intel','Opportunity Sizing']], drop_first=True)
df1.head()


# In[34]:


df2=df.copy()
df2 = pd.concat([df2, df1], axis=1)
df2.head()


# In[35]:


df2 = df2.drop(['Technology', 'City', 'Business Sales Medium', 'Client Revenue Sizing',
              'Client Employee Sizing', 'Business from Client Last Year',
              'Compete Intel', 'Opportunity Sizing'], axis = 1)
df2.head()


# In[36]:


df2['Opportunity Status']=df2['Opportunity Status'].map({'Won':1,'Loss':0})


# In[37]:


df2


# 1.Most of the Sales Medium are via MArketing and Enterprise Sellers</br>
# 2.

# In[38]:


X=df2.drop('Opportunity Status', axis=1)
y = df2['Opportunity Status']


# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3, random_state=125)


# In[40]:


from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X_train[['Sales Velocity','Sales Stage','Opportunity Size']] = scaler.fit_transform(X_train[['Sales Velocity','Sales Stage','Opportunity Size']])


# In[41]:


X_test[['Sales Velocity','Sales Stage','Opportunity Size']] = scaler.fit_transform(X_test[['Sales Velocity','Sales Stage','Opportunity Size']])


# In[42]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
model=DecisionTreeClassifier()


# In[43]:


# fit the model with the training data
model.fit(X_train, y_train)


# In[44]:


# predict the target on the train dataset
predict_train = model.predict(X_train)
predict_train


# In[45]:


acc=accuracy_score(y_train, predict_train)
print('accuracy_score on train dataset : ', acc)


# In[46]:


from sklearn import metrics
# Confusion matrix 
confusion = metrics.confusion_matrix(y_train, predict_train )
print(confusion)


# In[47]:


from sklearn.metrics import precision_score, recall_score
precision_score(y_train,predict_train)


# In[48]:


recall_score(y_train,predict_train)


# In[49]:


predict_test = model.predict(X_test)
print('Target on test data\n\n',predict_test)


# In[50]:


confusion2 = metrics.confusion_matrix(y_test, predict_test )
print(confusion2)


# In[51]:


testaccuracy= accuracy_score(y_test,predict_test)
testaccuracy


# ### Grid Search CV

# In[52]:


from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
param_dist = {"max_depth": [3, None],
              "max_features": randint(1,50),
              "min_samples_leaf": randint(50,1000),
              "criterion": ["gini", "entropy"]}
#Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()
#Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)
#Fit it to the data
tree_cv.fit(X_train, y_train)
#Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))


# In[ ]:


from math import log2
from numpy import round

def b(p):
    if p == 0 or p == 1:
        return 0
    
    q = 1 - p
    return -(p*log2(p) + q*log2(q))

def entropy(column):
    p = column.value_counts()[0] / column.count()
    return b(p)

initial_entropy = entropy(df["Opportunity Status"])
for name in list(features):
    number_of_values = data_labeled[name].nunique()
    
    if number_of_values == 2:
        df_true = data_labeled[data_labeled[name] == 0]
        df_false = data_labeled[data_labeled[name] == 1]
        
        p = df_true.shape[0] / (data_labeled.shape[0])
        q = 1 - p
        
        entropy_true = entropy(df_true["Churn Status"])
        entropy_false = entropy(df_false["Churn Status"])
        
        gain = initial_entropy - (p*entropy_true + q*entropy_false)
        gain = round(gain, 2)
        
        print(f"{name} has an information gain of {round(gain, 2)}")
        if gain < 0.01:
            features.remove(name)


# In[56]:


df.head()


# In[1]:


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV


# ### Finding Best Depth

# In[61]:


for i in range(1,10):
    dt=DecisionTreeClassifier(max_depth=i)
    dt.fit(X_train,y_train)
    training_accuracy=accuracy_score(y_train,dt.predict(X_train))
    val_accuracy=cross_val_score(dt,X_train,y_train,cv=10)
    print('Depth:',i, 'Training accuracy:', training_accuracy, "Cross val score:", np.mean(val_accuracy))
     


# ### gini index

# In[62]:


clf_gini = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=0)
clf_gini.fit(X_train, y_train)


# In[63]:


y_pred_gini = clf_gini.predict(X_test)


# In[64]:


print('Model accuracy score with criterion gini index: {0:0.4f}'. format(accuracy_score(y_test, y_pred_gini)))


# In[65]:


print('Training set score: {:.4f}'.format(clf_gini.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_gini.score(X_test, y_test)))


# ### criterion entropy

# In[66]:


clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=0)


# fit the model
clf_en.fit(X_train, y_train)


# In[67]:


y_pred_en = clf_en.predict(X_test)


# In[68]:


print('Model accuracy score with criterion entropy: {0:0.4f}'. format(accuracy_score(y_test, y_pred_en)))


# In[69]:


y_pred_train_en = clf_en.predict(X_train)

y_pred_train_en


# In[70]:


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train_en)))


# In[71]:



print('Training set score: {:.4f}'.format(clf_en.score(X_train, y_train)))

print('Test set score: {:.4f}'.format(clf_en.score(X_test, y_test)))


# In[72]:


print(classification_report(y_test, y_pred_en))


# In[2]:





# In[ ]:




