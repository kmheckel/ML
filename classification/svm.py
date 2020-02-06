import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# for some reason SVM is just trash...
clf = svm.SVC()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
print("accuracy = " + str(accuracy))

example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,2,2,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures),-1)

prediction = clf.predict(example_measures)
print(prediction)
