# EX 02: Developing a Neural Network Classification Model

## AIM

To develop a neural network classification model for the given dataset.

## Problem Statement

An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
![NNMODEL](https://github.com/user-attachments/assets/94915ce6-608d-45ca-83e6-c0e0c0a6c873)
## DESIGN STEPS

### STEP 1:
Import the packages and reading the dataset.
### STEP 2:
Preprocess and split the data.
### STEP 3:
Create a model
### STEP 4:
Plot the Training Loss, Validation Loss Vs Iteration Plot.

## PROGRAM

### Name: Adhithya M R
### Register Number: 212222240002

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import tensorflow as tf
import seaborn as sns
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import classification_report,confusion_matrix
import numpy as np
import matplotlib.pylab as plt

customer_df = pd.read_csv('customers.csv')
customer_df.head()

## Data Exploration

customer_df.columns

customer_df.dtypes

customer_df.shape

customer_df.isnull().sum()

customer_df_cleaned = customer_df.dropna(axis=0)

customer_df_cleaned.isnull().sum()

customer_df_cleaned.shape

customer_df_cleaned.dtypes

customer_df_cleaned['Gender'].unique()

customer_df_cleaned['Ever_Married'].unique()

customer_df_cleaned['Graduated'].unique()

customer_df_cleaned['Profession'].unique()

customer_df_cleaned['Spending_Score'].unique()

customer_df_cleaned['Var_1'].unique()

customer_df_cleaned['Segmentation'].unique()

categories_list=[['Male', 'Female'],
           ['No', 'Yes'],
           ['No', 'Yes'],
           ['Healthcare', 'Engineer', 'Lawyer', 'Artist', 'Doctor',
            'Homemaker', 'Entertainment', 'Marketing', 'Executive'],
           ['Low', 'Average', 'High']
           ]
enc = OrdinalEncoder(categories=categories_list)

cust_1=customer_df_cleaned.copy()
cust_1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']]=enc.fit_transform(cust_1[['Gender','Ever_Married','Graduated','Profession','Spending_Score']])

cust_1.dtypes

le = LabelEncoder()
cust_1['Segmentation'] = le.fit_transform(cust_1['Segmentation'])
cust_1.dtypes

cust_1 = cust_1.drop('ID',axis=1)
cust_1 = cust_1.drop('Var_1',axis=1)
cust_1.dtypes

# Calculate the correlation matrix
corr = cust_1.corr()

# Plot the heatmap
sns.heatmap(corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="BuPu",
        annot= True)
print('Adhithya M R 212222240002')

sns.pairplot(cust_1)
sns.distplot(cust_1['Age'])

plt.figure(figsize=(10,6))
sns.countplot(cust_1['Family_Size'])

plt.figure(figsize=(10,6))
sns.boxplot(x='Family_Size',y='Age',data=cust_1)

plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Spending_Score',data=cust_1)

plt.figure(figsize=(10,6))
sns.scatterplot(x='Family_Size',y='Age',data=cust_1)

cust_1.describe()
cust_1['Segmentation'].unique()

X=cust_1[['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size']].values

y1 = cust_1[['Segmentation']].values
one_hot_enc = OneHotEncoder()
one_hot_enc.fit(y1)
y1.shape
y = one_hot_enc.transform(y1).toarray()
y.shape
y1[0]
y[0]
X.shape
X_train,X_test,y_train,y_test=train_test_split(X,y,
                                               test_size=0.33,
                                               random_state=50)

X_train[0]
X_train.shape
scaler_age = MinMaxScaler()
scaler_age.fit(X_train[:,2].reshape(-1,1))
X_train_scaled = np.copy(X_train)
X_test_scaled = np.copy(X_test)

# To scale the Age column
X_train_scaled[:,2] = scaler_age.transform(X_train[:,2].reshape(-1,1)).reshape(-1)
X_test_scaled[:,2] = scaler_age.transform(X_test[:,2].reshape(-1,1)).reshape(-1)

# Creating the model
ai_brain = Sequential([Dense(6,activation='relu',input_shape=[8]),Dense(10,activation='relu'),
                  Dense(10,activation='relu'),Dense(4,activation='softmax')])

ai_brain.compile(optimizer='adam',
                 loss='categorical_crossentropy', # choose your loss function,
                 metrics=['accuracy'])

#early_stop = EarlyStopping(monitor='val_loss', patience=2)
early_stop = EarlyStopping(monitor='val_loss', patience=2)

ai_brain.fit(x=X_train_scaled,y=y_train,
             epochs=2000,
             batch_size=256,
             validation_data=(X_test_scaled,y_test),
             )

metrics = pd.DataFrame(ai_brain.history.history)

metrics.head()

print('Adhithya M R 212222240002')
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(ai_brain.predict(X_test_scaled), axis=1)

x_test_predictions.shape

y_test_truevalue = np.argmax(y_test,axis=1)

y_test_truevalue.shape

print(confusion_matrix(y_test_truevalue,x_test_predictions))
print('Adhithya M R 212222240002')
print(classification_report(y_test_truevalue,x_test_predictions))
print('Adhithya M R 212222240002')

# Saving the Model
ai_brain.save('customer_classification_model.h5')

# Saving the data
with open('customer_data.pickle', 'wb') as fh:
   pickle.dump([X_train_scaled,y_train,X_test_scaled,y_test,cust_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le], fh)

# Loading the Model
ai_brain = load_model('customer_classification_model.h5')

# Loading the data
with open('customer_data.pickle', 'rb') as fh:
   [X_train_scaled,y_train,X_test_scaled,y_test,cust_1,customer_df_cleaned,scaler_age,enc,one_hot_enc,le]=pickle.load(fh)



# Prediction for a single input
x_single_prediction = np.argmax(ai_brain.predict(X_test_scaled[1:2,:]), axis=1)

print('Adhithya M R 212222240002')
print(x_single_prediction)
print('Adhithya M R 212222240002')
print(le.inverse_transform(x_single_prediction))
```

## Dataset Information
![pic1](https://github.com/user-attachments/assets/7818f05a-82af-4ae8-a18c-9c9436d3b6bf)

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
![pic2](https://github.com/user-attachments/assets/6b1b2db1-0181-4cce-8a30-4a41599d1219)

### Classification Report
![pic3](https://github.com/user-attachments/assets/a7402a94-6c55-4829-af83-0d185b3f7381)

### Confusion Matrix
![pic4](https://github.com/user-attachments/assets/0328a174-23f8-42b8-88a6-db89eb465463)

### New Sample Data Prediction 
![pic5](https://github.com/user-attachments/assets/c0ae65f7-6aec-4d92-91fe-2b4092c9e91a)

![pic6](https://github.com/user-attachments/assets/6f0feeb0-cd75-4856-abc4-855c438afd89)

## RESULT
A neural network classification model for the given dataset is successfully developed. 
