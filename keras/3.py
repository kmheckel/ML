import tensorflow.keras as keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import pickle
import time

# values to try in grid-search
dense_layers = [0, 1, 2]
layer_sizes = [32, 64, 128]
conv_layers = [1, 2, 3]

# create a tensorboard callback to analyze the models
tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

# Load the dataset prepared by our pipeline
X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

#scale rgb values
X = X/255.0

#each loop allows us to grid search over our specified parameters
for dense_layer in dense_layers:
    for layer_size in layer_sizes:
        for conv_layer in conv_layers:
            # create a name for the model
            NAME = "{}-conv-{}-nodes-{}-dense-{}".format(conv_layer, layer_size, dense_layer, int(itme.time()))

            #our convnet is a sequential model
            model = Sequential()
            
            # we want at least one conv layer
            #the kernel/window size is 3x3
            # the shape[1:] gets the size of the input datapoint
            model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))
            model.add(Activation("relu"))
            #pooling window has size 2x2
            model.add(MaxPooling2D(pool_size=(2,2)))

            #Loop that optionally stacks additional conv layers
            for l in range(conv_layer-1):
                model.add(Conv2D(layer_size, (3,3), input_shape = X.shape[1:]))
                model.add(Activation("relu"))
                model.add(MaxPooling2D(pool_size=(2,2)))

            #flatten the model before passing to dense layers
            model.add(Flatten())

            #add dense layers as the search specifies
            for l in range(dense_layer):
                model.add(Dense(layer_size))
                model.add(Activation('relu'))

            # final classification layer that represents the decision between cat/dog
            model.add(Dense(1))
            model.add(Activation('sigmoid'))

            #compile the model
            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])
            # perform validation against test data. increase epochs for better trend information
            model.fit(X, y, batch_size=32, epochs=3, validation_split=0.1, callbacks=[tensorboard])

            #save each unique model so that you can reload it later.
            model.save(NAME)
