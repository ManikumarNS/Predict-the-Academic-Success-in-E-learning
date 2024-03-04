import pennylane as qml
import tensorflow as tf
import numpy as np
from Save_load import *
X_train = load("x_train")
Y_train = load("y_train")
X_test = load("x_test")
Y_test = load("y_test")
# Define a quantum circuit using PennyLane
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

# Train the model
model.fit(X, Y, epochs=400, batch_size=10)

y_predict = np.argmax(model.predict(X_test), axis=1)
print(y_predict)
