#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn


# In[2]:


data = pd.read_csv("Titanic-Dataset.csv")
data.head()


# In[3]:


data = data.drop("PassengerId", axis=1)


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.shape


# In[7]:


data.duplicated().sum()


# In[8]:


data.isna().sum().sort_values(ascending = False)


# In[9]:


data = data.drop("Cabin", axis=1)


# In[10]:


data["Embarked"] = data["Embarked"].fillna(data["Embarked"].mode()[0])


# In[11]:


data.isna().sum().sort_values(ascending = False)


# In[12]:


data.head()


# In[13]:


from sklearn.impute import KNNImputer


# In[14]:


imputer = KNNImputer(n_neighbors=4)


# In[15]:


data["Age"] = imputer.fit_transform(data[["Age"]])


# In[16]:


data.isna().sum().sort_values(ascending = False)


# In[17]:


data['Survived'].value_counts(normalize=True)*100


# In[18]:


sns.countplot(x = "Survived", data=data)
plt.title("Survival Count")
plt.xlabel("Survived(1 = Yes, 0 = No)")
plt.ylabel("Number of passenger")
plt.show()


# In[19]:


data.groupby("Sex")["Survived"].mean()*100


# In[20]:


sns.barplot(x='Sex', y='Survived', data=data)
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.show()


# In[21]:


data.groupby("Pclass")["Survived"].mean()*100


# In[22]:


sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title('Survival Count by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Number of Passengers')
plt.show()


# In[23]:


data.groupby("Survived")["Age"].mean()


# In[24]:


sns.histplot(data["Age"], bins=30, kde=True)
plt.title("Age Distribution of Titanic Passenger")
plt.xlabel("Age")
plt.ylabel("Number of Passenger")
plt.show()


# In[25]:


plt.figure(figsize=(10,6))
sns.histplot(data=data, x='Age', hue='Survived', multiple='stack', bins=30, kde=True)
plt.title("Age Distribution of Survivors vs Non-Survivors")
plt.xlabel("Age")
plt.ylabel("Count")
plt.legend(labels=["Did Not Survive", "Survived"])
plt.show()


# In[26]:


data.groupby(["Pclass", "Survived"])["Fare"].mean()


# In[27]:


plt.figure(figsize=(10,6))
sns.boxplot(x='Pclass', y='Fare', hue='Survived', data=data)
plt.title("Fare Variation Across Classes and Survival Status")
plt.xlabel("Passenger Class")
plt.ylabel("Fare")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()


# In[28]:


data.groupby("Embarked")["Survived"].mean()*100


# In[29]:


plt.figure(figsize=(8,5))
sns.countplot(x='Embarked', hue='Survived', data=data)
plt.title("Survival Count by Embarkation Port")
plt.xlabel("Embarked Port (S = Southampton, C = Cherbourg, Q = Queenstown)")
plt.ylabel("Number of Passengers")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.show()


# In[30]:


data["FamilySize"] = data["SibSp"] + data["Parch"]


# In[31]:


sns.countplot(x="FamilySize", hue="Survived", data=data)
plt.title("Survival by Family Size")
plt.xlabel("Family Size")
plt.ylabel("Numbeeer of Passenger")
plt.legend(title="Survived", labels=["Yes", "No"])
plt.show()


# In[32]:


numeric_data = data[['Survived', 'Age', 'Fare', 'SibSp', 'Parch']]
correlation = numeric_data.corr()


# In[33]:


plt.figure(figsize=(8,6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Between Numerical Features")
plt.show()


# In[67]:


X = data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize']]
y = data['Survived']


# In[68]:


from sklearn.preprocessing import LabelEncoder


# In[69]:


columns_to_encode = ["Sex", "Embarked"]


# In[70]:


label_enocder = {}


# In[71]:


for col in columns_to_encode:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_enocder[col] = le


# In[72]:


X_train["Sex"] = le.fit_transform(X_train["Sex"])


# In[73]:


data.head()


# In[75]:


from sklearn.model_selection import train_test_split


# In[76]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


# In[77]:


from sklearn.ensemble import RandomForestClassifier


# In[78]:


model = RandomForestClassifier(random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
# In[82]:


y_pred = model.predict(X_test)


# In[85]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[87]:


print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[89]:


data.to_csv("cleaned_titanic.csv", index=False)

