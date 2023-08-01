import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
'''a (1) a 4-1-2 network,'''
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

# Filter data for class Iris-versicolor and Iris-virginica
X_filtered = X[(y == 'Iris-versicolor') | (y == 'Iris-virginica')]
y_filtered = y[(y == 'Iris-versicolor') | (y == 'Iris-virginica')]

# Convert the target variable to one-hot encoded vectors (class 2 and 3 only)
y_one_hot = np.zeros((y_filtered.shape[0], 2))
for i in range(y_filtered.shape[0]):
    if y_filtered[i] == 'Iris-versicolor':
        y_one_hot[i, 0] = 1
    elif y_filtered[i] == 'Iris-virginica':
        y_one_hot[i, 1] = 1

# Split the filtered data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_one_hot, test_size=0.2, random_state=4)



learning_rate = 0.1
iterations = 1000
N = y_train.size

# Input features
input_size = 4

# Hidden layers
hidden_size = 1

# Output layer
output_size = 2

results = pd.DataFrame(columns=["mse", "accuracy"])

np.random.seed(10)

# Hidden layer
W1 = np.random.normal(scale=0.5, size=(input_size, hidden_size))

# Output layer
W2 = np.random.normal(scale=0.5, size=(hidden_size, output_size))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def mean_squared_error(y_pred, y_true):
    return ((y_pred - y_true)**2).sum() / (2 * y_pred.shape[0])

def accuracy(y_pred, y_true):
    acc = y_pred.argmax(axis=1) == y_true.argmax(axis=1)
    return acc.mean()

for itr in range(iterations):
    # Implementing feedforward propagation on hidden layer
    Z1 = np.dot(X_train, W1)
    A1 = sigmoid(Z1)

    # Implementing feed forward propagation on output layer
    Z2 = np.dot(A1, W2)
    A2 = sigmoid(Z2)

    # Calculating the error
    mse = mean_squared_error(A2, y_train)
    acc = accuracy(A2, y_train)
    results = results.append({"mse": mse, "accuracy": acc}, ignore_index=True)

    # Backpropagation phase
    E1 = A2 - y_train
    dW1 = E1 * A2 * (1 - A2)

    E2 = np.dot(dW1, W2.T)
    dW2 = E2 * A1 * (1 - A1)

    # Updating the weights
    W2_update = np.dot(A1.T, dW1) / N
    W1_update = np.dot(X_train.T, dW2) / N

    W2 = W2 - learning_rate * W2_update
    W1 = W1 - learning_rate * W1_update

plt.plot(results["mse"], label='Mean Squared Error')
plt.plot(results["accuracy"], label='Accuracy')
plt.title('Training Progress')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.legend()
plt.show()

# Testing phase
Z1 = np.dot(X_test, W1)
A1 = sigmoid(Z1)

Z2 = np.dot(A1, W2)
A2 = sigmoid(Z2)

acc = accuracy(A2, y_test)
print("Accuracy: {}".format(acc))
final_mse = results["mse"].iloc[-1]
print("Training Loss: {}".format(final_mse))

def predict(X, W1, W2):
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

    # Get the index of the maximum value in each row
    predictions = np.argmax(A2, axis=1)

    label_to_name = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
    flower_names = np.array([label_to_name[label] for label in predictions])

    return predictions, flower_names

# Load new data
new_data = np.array([[5.3, 3.7, 1.5, 0.2], [7.0, 3.2, 4.7, 1.4]])

# Make predictions on new data
predictions = predict(new_data, W1, W2)

# Print the predicted class labels
print(predictions)
