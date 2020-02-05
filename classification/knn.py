import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

#read in the csv file
df=pd.read_csv('breast-cancer-wisconsin.data')

#set the missing values as outliers. dropping anything with missing values could remove too much valuable data
# dropping also could be an option since only a small fraction have missing values
df.replace('?', -99999, inplace=True)

# get rid of the id column since it has no bearing on cancer predictions
df.drop(['id'],1,inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


