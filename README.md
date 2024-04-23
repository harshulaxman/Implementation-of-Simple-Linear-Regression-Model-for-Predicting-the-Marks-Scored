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
![image](https://github.com/harshulaxman/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145686689/84a0d8f1-c336-4023-b031-0b1516bb6d40)

![image](https://github.com/harshulaxman/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145686689/bf3a21a3-251e-468e-8ccc-5a67048f64b2)

![image](https://github.com/harshulaxman/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145686689/b567f9df-b60e-47e4-b883-8226e4c1ade0)
![image](https://github.com/harshulaxman/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145686689/c999e140-ff32-4940-9d31-2212e329a118)
![image](https://github.com/harshulaxman/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145686689/0b7fce43-2c18-4313-9e44-ac458dd0f47e)
![image](https://github.com/harshulaxman/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145686689/1f796af4-2234-4a6d-b593-efa2de219762)




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using Python programming.
