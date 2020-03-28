#this file covers importing images of various sizes and converting them
# for use in a convolutional neural network

import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle



DATADIR = "../../Downloads/PetImages" #path to data directory

CATEGORIES = ["Dog", "Cat"] # name of class directories

#list/array to load data into
training_data = []

#set the size for images to be resized to
IMG_SIZE = 128

def create_training_data():
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category) #get path to data directories
        #set the label as the index from category array
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path): # each image in data directory
            #load each image at the path target as grayscale
            try:
                #load image
                img_array = cv2.imread(os.path.join(path,img), cv2.IMREAD_GRAYSCALE)
                #resize image
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                #append image to data array
                training_data.append([new_array, class_num])
            except Exception as e:
                pass #catch exceptions from bad images

# call the function
create_training_data()

#plt.imshow(img_array, cmap="gray")


#shuffle the data so that the model doesn't guess all dogs then all cats
random.shuffle(training_data)

X = []
y = []
# place loaded data into arrays
for features, label in training_data:
    X.append(features)
    y.append(label)

#reshape array for neural net
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1) # the one represents the grayscale, 3 would be for coloor


# how to save the training data to be loaded back at a later time:
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

#how to open up/load previous parsed data.
#pickle_in = open("X.pickle", "rb")
#X = pickle.load(pickle_in)
