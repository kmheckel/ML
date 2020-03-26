import tensorflow as tf
import tensorflow.keras as keras 
import matplotlib.pyplot as plt
import numpy as np

#load the mnist dataset
mnist = tf.keras.datasets.mnist # 28x28 images of hadnwritten digits

#split data into training and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize the data to between 0 and 1
x_train = keras.utils.normalize(x_train, axis=1)
x_test = keras.utils.normalize(x_test, axis=1)

#build the model one layer at a time:

#we use sequential to make a feedforward network
model = keras.models.Sequential()

#flatten the input to a 1x784 vector for layer one
model.add(keras.layers.Flatten())

#add two fully connected hidden layers with 128 neurons each
#and an activation function of RELU
model.add(keras.layers.Dense(128, activation=tf.nn.relu))
model.add(keras.layers.Dense(128, activation=tf.nn.relu))

#Add the output layer which will have as many neurons as classes you
#are trying to predict. the softmax activation function is used
#to get a probability distribution for predictions
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

#Build the model with parameters including the optimization function 
#(could also use stochastic gradient descent), the loss function to optimize against,
#and metrics to report after every epoch
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#Train the model
model.fit(x_train, y_train, epochs=3)

#Test the model against the test set
val_loss, val_accuracy = model.evaluate(x_test, y_test)

#print results
print(val_loss, val_accuracy)

#save the model for future use
model.save('my_first_model')

#load an old model
new_model = keras.models.load_model('my_first_model')

#view class probabilities. takes nparray as argument
predictions = new_model.predict(x_test)

#print(predictions)

#print the class precition and graph the result.
print(np.argmax(predictions[0]))
plt.imshow(x_test[0])
plt.show()
