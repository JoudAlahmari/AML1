# -*- coding: utf-8 -*-
"""
joud alahmari 2008071
"""
# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#1.import data set
dataset = pd.read_csv('DataLab1.csv')

#2. Separate the Independent and dependent variables.
#extract the independent variables -function of the Pandas library- .
#extract selected rows and columns from the dataset.
X= dataset.iloc[:,:-1].values  
y= dataset.iloc[:,3].values  

#3. Calculate the median for each feature or column that contains a missing value and replace the result for the missing value.
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy='median')
imputer = imputer.fit(X[:, 1:4])
X[:, 1:4] = imputer.transform(X[:, 1:4])


# 4.Encoding the categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder ='passthrough')
X = np.array(ct.fit_transform(X), dtype= float)
y = LabelEncoder().fit_transform(y)

#5. Splitting the dataset into the Training set and Test Set and assign 30% for test set. (random_state = 0).
from sklearn.model_selection import train_test_split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=0)

 
#6.Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)





























