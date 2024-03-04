from Save_load import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dropout, Dense, MaxPooling2D
from tensorflow.keras.datasets import mnist
import numpy as np
from confusion_matrix import confu_matrix,multi_confu_matrix
from keras import regularizers
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

X_train = load("x_train")
Y_train = load("y_train")
X_test = load("x_test")
Y_test = load("y_test")

import pennylane as qml
import tensorflow as tf
import numpy as np
from Save_load import *


def quantum_circuit(params):
    qml.Hadamard(wires=0)
    qml.RY(params[0], wires=0)
    return qml.expval(qml.PauliZ(0))

# Create a PennyLane device
dev = qml.device("default.qubit", wires=1)

# Define a QNode for the quantum circuit
@qml.qnode(dev)
def qnode(params):
    return quantum_circuit(params)

# Define a classical neural network model using TensorFlow/Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation='relu', input_shape=(X_train[1].shape)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Combine the quantum circuit and classical neural network
def quantum_neural_network(params):
    return model(np.array([qnode(params).numpy()]))


# Generate some random data
X = X_train
Y = Y_train
# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=5000, batch_size=100, verbose=0)
y_predict = np.argmax(model.predict(X_test), axis=1)
#return y_predict, confu_matrix(Y_test, y_predict)

def cnn(X_train,Y_train,X_test,Y_test):

    # reshaping data
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1, 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1, 1))


    model = Sequential()
    model.add(Conv2D(64, (1, 1), padding='valid', input_shape=X_train[1].shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(1, 1)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=1000, batch_size=10, verbose=1)
    y_predict = np.argmax(model.predict(X_test), axis=1)
    return y_predict, #confu_matrix(Y_test, y_predict)
#cnn(X_train,Y_train,X_test,Y_test)


def stacked_autoencoder(X_train,Y_train,X_test,Y_test):

    model = Sequential()
    model.add(Dense(20, activation='softmax'))
    model.add(Dense(10, activation='softmax'))
    model.add(Dense(3, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=2000, batch_size=10, verbose=0)
    y_predict = np.argmax(model.predict(X_test), axis=1)
    y_predict_train = np.argmax(model.predict(X_train), axis=1)
    return y_predict,y_predict_train, confu_matrix(Y_test, y_predict)

stacked_autoencoder(X_train,Y_train,X_test,Y_test)

def pro_classifers(X_train, Y_train, X_test, Y_test):

    pred_1 = cnn(X_train, Y_train, X_test, Y_test)

    pred_2 = stacked_autoencoder(X_train, Y_train, X_test, Y_test)

    predict = np.mean((pred_1,pred_2), axis=0)

    predict = np.round(predict)

    met = multi_confu_matrix(Y_test,predict)

    return met