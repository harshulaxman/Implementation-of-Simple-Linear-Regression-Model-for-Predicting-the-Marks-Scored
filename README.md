# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn. 
4. Assign the points for representing in the graph.
5. Predict the regression for marks by using the representation of the graph. 
6. Compare the graphs and hence we obtained the linear regression for the given data. 

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: HARSSHITHA LAKSHMANAN
RegisterNumber: 212223230075

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/exp1data - Sheet1.csv')
df.head(10)
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(xtrain,ytrain)
xtrain
ytrain
lr.predict(xtest.iloc[0].values.reshape(1,1))
plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.plot(xtrain,lr.predict(xtrain),color='red')

*/
```

## Output:
![image](https://github.com/harshulaxman/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145686689/471bf8a5-f503-43e6-bdc0-6c82084b364e)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using Python programming.
