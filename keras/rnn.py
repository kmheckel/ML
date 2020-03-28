import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

mnist= keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train=x_train/255
x_test=x_test/255

model = Sequential()

#return sequences when next layer is recurrent, don't ret seqs when next layer is flat
model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(.2))

model.add(LSTM(128, activation='relu'))
model.add(Dropout(.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(.2))

model.add(Dense(10, activation='softmax'))

opt = keras.optimizers.Adam(lr=1e-3, decay=1e-5)

model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))

