import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'

# Load the data into a pandas dataframe
iris_df = pd.read_csv(url, header=None)

# Extract the features and target variable
X = iris_df.iloc[:, :-1].values
y = iris_df.iloc[:, -1].values

# Convert the target variable to one-hot encoded vectors
y_one_hot = np.zeros((y.shape[0], 3))
for i in range(y.shape[0]):
    if y[i] == 'Iris-versicolor':
        y_one_hot[i, 1] = 1
    else:
        y_one_hot[i, 2] = 1

X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=4)

learning_rate = 0.1
iterations = 1000
N = y_train.size

# Input features
input_size = 4

# Hidden layers
hidden_size1 = 2
hidden_size2 = 2

# Output layer
output_size = 3

results = pd.DataFrame(columns=["mse", "accuracy"])

np.random.seed(10)

# Hidden layer 1
W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size1))

# Hidden layer 2
W2 = np.random.normal(scale=0.5, size=(hidden_size1, hidden_size2))

# Output layer
W3 = np.random.normal(scale=0.5, size=(hidden_size2, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true)**2).sum() / (2*y_pred.shape[0])

def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()

for itr in range(iterations):

    # Implementing feedforward propagation on hidden layer 1
    Z1 = np.dot(X_train, W1)
    A1 = sigmoid(Z1)

    # Implementing feedforward propagation on hidden layer 2
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)

    # Implementing feedforward propagation on output layer
    Z3 = np.dot(A2, W3)
    A3 = sigmoid(Z3)

    # Calculating the error
    mse = mean_squared_error(A3, y_train)
    acc = accuracy(A3, y_train)
    results=results.append({"mse":mse, "accuracy":acc},ignore_index=True )

    # Backpropagation phase
    E3 = (A3 - y_train) * A3 * (1 - A3)
    dW3 = np.dot(A2.T, E3) / N

    E2 = np.dot(E3, W3.T) * A2 * (1 - A2)
    dW2 = np.dot(A1.T, E2) / N

    E1 = np.dot(E2, W2.T) * A1 * (1 - A1)
    dW1 = np.dot(X_train.T, E1) / N


    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2
    W3 -= learning_rate * dW3

plt.plot(results.mse, label='Training Loss')
plt.plot(results.accuracy, label='Testing Loss')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
print(acc)
print(mse)

def predict(X, W1, W2, W3):
    """
    Makes predictions using a trained neural network model.
    
    Arguments:
    X -- input data (n_samples, n_features)
    W1 -- weights of the hidden layer (n_features, n_hidden_units)
    W2 -- weights of the output layer (n_hidden_units, n_classes)
    
    Returns:
    predictions -- predicted class labels (n_samples, )
    """
    # Implementing feedforward propagation on hidden layer
    Z1 = np.dot(X, W1)
    A1 = sigmoid(Z1)

    # Implementing feed forward propagation on output layer
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)
    # Implementing feedforward propagation on output layer
    Z3 = np.dot(A2, W3)
    A3 = sigmoid(Z3)
    # Get the index of the maximum value in each row
    predictions = np.argmax(A3, axis=1)
    
    label_to_name = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    flower_names = np.array([label_to_name[label] for label in predictions])
    
    return predictions, flower_names
# Load new data
new_data = np.array([[5.3,3.7,1.5,0.2],[7.0,3.2,4.7,1.4]])

# Make predictions on new data
predictions = predict(new_data, W1, W2, W3)

# Print the predicted class labels
print(predictions)