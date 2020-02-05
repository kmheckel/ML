import pandas as pd
import numpy as np
import quandl, math, datetime
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

# Get stock data from quandl
df = quandl.get("WIKI/GOOGL", api_key='oH7knqk13pDCADDcQZsx')

# Calculate some custom features
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

# Create a new dataframe holding the features we want to use
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

# Select the column we would like to forecast
forecast_col = 'Adj. Close'

# Fill missing data with clear outliers to avoid influencing data
df.fillna(-99999, inplace=True)

#set the distance into the future we want to predict
forecast_out = int(math.ceil(0.01 * len(df)))

# Create the Truth column
df['label'] = df[forecast_col].shift(-forecast_out)


# X is the array containing all of our data features
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X = X[:-forecast_out] # 
X_lately = X[-forecast_out:]


# Drop any records missing info
df.dropna(inplace=True)

#y contains our labels / Truth values
y = np.array(df['label'])

# This is optional based on practicality. Scales the feautres to make the model work better.

# Split the data into training and test sets
# 20% of the data is set aside as testing data. We don't train on the testing data so that we
# can effectively test our model for overfitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# instantiate a classifer
clf = LinearRegression()

#Uncomment the following line to switch to SVM:
# The SVM is much worse on this data than linear regression.
#clf = svm.SVR()

# Train / fit the model to the training data
clf.fit(X_train, y_train)

# Evaluate our model against the test data we set aside and print the accuracy.
accuracy = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
  next_date = datetime.datetime.fromtimestamp(next_unix)
  next_unix += one_day
  df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
