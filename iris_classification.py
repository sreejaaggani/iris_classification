#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install pandas')


# In[4]:


get_ipython().system('pip install numpy')


# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# In[7]:


get_ipython().system('pip install matplotlib.pyplt')


# In[8]:


get_ipython().system('pip install matplotlib.pyplot')


# In[9]:


get_ipython().system('pip install seaborn')


# In[73]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv(r'C:\Users\Sreeja Aggani\Downloads\Iris (1).csv')


# In[4]:


df.shape#data points and features


# In[5]:


df.columns


# In[6]:


df["Species"].value_counts()


# In[7]:


df.head()


# In[8]:


df.describe()


# In[9]:


df.info()#to display basic details of info


# In[10]:


#Preprocessing the data
#1.check the null values
df.isnull().sum()


# In[11]:


#Exploratory Data analysis
df['SepalLengthCm'].hist()#gaussian curve we get


# In[12]:


#Scatter plot
colors = ['blue','orange','green']
Species = ['Iris-setosa', 
'Iris-versicolor',
'Iris-virginica'  ]


# In[13]:


for i in range(3): #in three species drawing the line y=mx+c
    x = df[df['Species']==Species[i]]#Linearly separable
    plt.scatter(x['SepalLengthCm'],x['SepalWidthCm'],c = colors[i],label=Species[i])
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.legend()


# In[14]:


for i in range(3): #in three species
    x = df[df['Species']==Species[i]]
    plt.scatter(x['PetalLengthCm'],x['PetalWidthCm'],c = colors[i],label=Species[i])
    plt.xlabel("petal Length")
    plt.ylabel("petal Width")
    plt.legend()


# In[15]:


for i in range(3): #in three species
    x = df[df['Species']==Species[i]]
    plt.scatter(x['SepalLengthCm'],x['PetalLengthCm'],c = colors[i],label=Species[i])
    plt.xlabel("sepal Length")
    plt.ylabel("petal length")
    plt.legend()


# In[16]:


for i in range(3): #in three species
    x = df[df['Species']==Species[i]]
    plt.scatter(x['SepalWidthCm'],x['PetalWidthCm'],c = colors[i],label=Species[i])
    plt.xlabel("sepal Length")
    plt.ylabel("petal length")
    plt.legend()


# In[17]:


#pair plot4c2 = 6
plt.close();
sns.set_style("whitegrid")
sns.pairplot(df, hue = 'Species',size = 3);
plt.show()
#pair plots are useful only for small dataset
#petal length and petal width are more imp to classify


# In[23]:


df.corr()


# In[24]:


corr = df.corr()
fig, ax = plt.subplots(figsize = (3,4))
sns.heatmap(corr,annot = True,ax = ax)


# In[35]:


pip install scikit-learn


# In[61]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()



# In[62]:


df['Species'] = le.fit_transform(df['Species'])
df.head()


# In[63]:


#Model trainng
from sklearn.model_selection import train_test_split
x = df.drop(columns = ['Species'])
y = df['Species']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.30)


# In[64]:


#Logistic regession
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[65]:


model.fit(x_train, y_train)


# In[66]:


#print the performance
print("Accuracy",model.score(x_test,y_test)*100)


# In[67]:


#KNN-
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[68]:


model.fit(x_train, y_train)


# In[69]:


#print metric to get performance
print("Accuracy",model.score(x_test,y_test)*100)


# In[70]:


#Decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()


# In[71]:


model.fit(x_train, y_train)


# In[72]:


print("Accuracy",model.score(x_test,y_test)*100)


# In[ ]:




