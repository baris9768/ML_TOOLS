# Machine Learning Tools Library

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

# Function to load data
def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded data.
    """
    return pd.read_csv(file_path)

# Function to preprocess data
def preprocess_data(data):
    """
    Preprocess the data by filling missing values, dropping columns with NaN values
    and zero standard deviation, and scaling the features.
    
    Args:
        data (pd.DataFrame): The data to preprocess.
        
    Returns:
        np.ndarray: Preprocessed data.
    """
    data.fillna(method='ffill', inplace=True)
    data.dropna(axis=1, inplace=True)
    data = data.loc[:, data.std() != 0]
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

# Function to split data
def split_data(data, target_column):
    """
    Split the data into training and testing sets.
    
    Args:
        data (pd.DataFrame): The data to split.
        target_column (str): The name of the target column.
        
    Returns:
        tuple: Training and testing sets (X_train, X_test, y_train, y_test).
    """
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Function for visualization
def visualize_data(data):
    """
    Visualize the data using pairplot.
    
    Args:
        data (pd.DataFrame): The data to visualize.
    """
    sns.pairplot(data)
    plt.show()

# Function for linear regression
def linear_regression(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a linear regression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Linear Regression R^2 Score:", model.score(X_test, y_test))

# Function for logistic regression
def logistic_regression(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a logistic regression model.
    """
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Function for decision tree classifier
def decision_tree(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a decision tree classifier.
    """
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Function for random forest classifier
def random_forest(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a random forest classifier.
    """
    model = RandomForestClassifier(n_estimators=100)  # n_estimators can be tweaked
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Function for support vector machine
def support_vector_machine(X_train, y_train, X_test, y_test):
    """
    Train and evaluate a support vector machine classifier.
    
    Important Parameters:
    - kernel: Can be 'linear', 'rbf', 'poly', etc.
    - C: Regularization parameter.
    """
    model = SVC(kernel='linear', C=1.0)  # kernel and C can be tweaked
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Function for artificial neural network
def artificial_neural_network(X_train, y_train, X_test, y_test):
    """
    Train and evaluate an artificial neural network.
    
    Important Parameters:
    - layers: Number and size of hidden layers.
    - activation: Activation functions for each layer.
    - epochs: Number of epochs to train the model.
    - batch_size: Size of the batches for training.
    """
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(1, activation='sigmoid')  # Use 'softmax' for multi-class classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

    # Plot Loss and Accuracy
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Function for convolutional neural network
def convolutional_neural_network(X_train, y_train, X_test, y_test, img_height, img_width):
    """
    Train and evaluate a convolutional neural network for image data.
    
    Important Parameters:
    - Conv2D: Number of filters and kernel size.
    - MaxPooling2D: Pool size.
    - Dense: Number of neurons and activation functions.
    - epochs: Number of epochs to train the model.
    - batch_size: Size of the batches for training.
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10, activation='softmax')  # Adjust number of neurons for number of classes
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Function for k-means clustering
def kmeans_clustering(data):
    """
    Perform k-means clustering on the data.
    
    Important Parameters:
    - n_clusters: Number of clusters.
    """
    kmeans = KMeans(n_clusters=3)  # n_clusters can be tweaked
    kmeans.fit(data)
    labels = kmeans.labels_
    return labels

# Function for principal component analysis
def pca_analysis(data):
    """
    Perform principal component analysis (PCA) on the data.
    
    Important Parameters:
    - n_components: Number of principal components.
    """
    pca = PCA(n_components=2)  # n_components can be tweaked
    principal_components = pca.fit_transform(data)
    return principal_components

# Function for the E-step in EM algorithm
def e_step(X, weights, means, covariances):
    """
    Perform the E-step in the EM algorithm.
    
    Args:
        X (np.ndarray): Input data.
        weights (np.ndarray): Weights of the mixture components.
        means (np.ndarray): Means of the mixture components.
        covariances (np.ndarray): Covariance matrices of the mixture components.
        
    Returns:
        np.ndarray: Responsibilities of the mixture components for each data point.
    """
    n_samples, n_components = X.shape[0], weights.shape[0]
    responsibilities = np.zeros((n_samples, n_components))
    
    for k in range(n_components):
        responsibilities[:, k] = weights[k] * multivariate_normal.pdf(X, mean=means[k], cov=covariances[k])
    
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities

# Function for the M-step in EM algorithm
def m_step(X, responsibilities):
    """
    Perform the M-step in the EM algorithm.
    
    Args:
        X (np.ndarray): Input data.
        responsibilities (np.ndarray): Responsibilities of the mixture components for each data point.
        
    Returns:
        tuple: Updated weights, means, and covariances of the mixture components.
    """
    n_samples, n_features = X.shape
    n_components = responsibilities.shape[1]
    
    weights = responsibilities.sum(axis=0) / n_samples
    means = np.dot(responsibilities.T, X) / responsibilities.sum(axis=0)[:, np.newaxis]
    covariances = np.zeros((n_components, n_features, n_features))
    
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(responsibilities[:, k] * diff.T, diff) / responsibilities[:, k].sum()
    
    return weights, means, covariances

# Function for cross-validation
def cross_validation(model, X, y):
    """
    Perform cross-validation on the given model.
    
    Important Parameters:
    - cv: Number of cross-validation folds.
    """
    scores = cross_val_score(model, X, y, cv=5)  # cv can be tweaked
    print("Cross-Validation Scores:", scores)

# Function for grid search hyperparameter tuning
def grid_search_tuning(X_train, y_train):
    """
    Perform grid search for hyperparameter tuning on a logistic regression model.
    
    Important Parameters:
    - param_grid: Dictionary with parameters names as keys and lists of parameter settings to try as values.
    """
    param_grid = {'C': [0.1, 1, 10], 'solver': ['lbfgs', 'liblinear']}  # param_grid can be tweaked
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_
    print("Best Parameters from Grid Search:", best_params)
    return best_params

# Function for basic neural network forward pass
def forward_pass(X, weights, biases):
    """
    Perform a forward pass through a basic neural network.
    
    Args:
        X (np.ndarray): Input data.
        weights (list): List of weights for each layer.
        biases (list): List of biases for each layer.
        
    Returns:
        np.ndarray: Output of the neural network.
    """
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    layer_input = X
    for W, b in zip(weights, biases):
        layer_input = sigmoid(np.dot(layer_input, W) + b)
    return layer_input

# Function for backpropagation
def backpropagation(X, y, weights, biases, learning_rate):
    """
    Perform backpropagation to update weights and biases.
    
    Args:
        X (np.ndarray): Input data.
        y (np.ndarray): Target values.
        weights (list): List of weights for each layer.
        biases (list): List of biases for each layer.
        learning_rate (float): Learning rate for gradient descent.
        
    Returns:
        tuple: Updated weights and biases.
    """
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    def sigmoid_derivative(z):
        return sigmoid(z) * (1 - sigmoid(z))
    
    # Forward pass
    layer_input = X
    activations = [X]
    zs = []
    for W, b in zip(weights, biases):
        z = np.dot(layer_input, W) + b
        zs.append(z)
        layer_input = sigmoid(z)
        activations.append(layer_input)
    
    # Backward pass
    delta = (activations[-1] - y) * sigmoid_derivative(zs[-1])
    nabla_w = [np.dot(activations[-2].T, delta)]
    nabla_b = [np.sum(delta, axis=0, keepdims=True)]
    
    for l in range(2, len(weights) + 1):
        z = zs[-l]
        sp = sigmoid_derivative(z)
        delta = np.dot(delta, weights[-l + 1].T) * sp
        nabla_w.insert(0, np.dot(activations[-l - 1].T, delta))
        nabla_b.insert(0, np.sum(delta, axis=0, keepdims=True))
    
    weights = [W - learning_rate * nw for W, nw in zip(weights, nabla_w)]
    biases = [b - learning_rate * nb for b, nb in zip(biases, nabla_b)]
    
    return weights, biases

# Custom neural network training function (Keras-like)
def train_custom_nn(X, y, layers, learning_rate=0.01, epochs=1000):
    """
    Train a custom neural network using forward pass, backpropagation, and gradient descent.
    
    Important Parameters:
    - layers: List of integers representing the number of neurons in each layer.
    - learning_rate: Learning rate for gradient descent.
    - epochs: Number of training epochs.
        
    Returns:
        tuple: Trained weights and biases.
    """
    # Initialize weights and biases
    weights = [np.random.randn(layers[i], layers[i+1]) for i in range(len(layers) - 1)]
    biases = [np.random.randn(1, layers[i+1]) for i in range(len(layers) - 1)]
    
    # Training loop
    for epoch in range(epochs):
        weights, biases = backpropagation(X, y, weights, biases, learning_rate)
        if epoch % 100 == 0:
            predictions = forward_pass(X, weights, biases)
            loss = np.mean((predictions - y) ** 2)
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights, biases

# Function for autoencoders
def autoencoder(X_train, X_test, encoding_dim):
    """
    Train and evaluate an autoencoder.
    
    Important Parameters:
    - encoding_dim: Dimension of the encoding layer.
    - epochs: Number of epochs to train the model.
    - batch_size: Size of the batches for training.
        
    Returns:
        tuple: Trained autoencoder model and encoder model.
    """
    input_dim = X_train.shape[1]
    
    # Define the autoencoder
    input_layer = tf.keras.layers.Input(shape=(input_dim,))
    encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
    decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)
    
    autoencoder = tf.keras.models.Model(input_layer, decoded)
    encoder = tf.keras.models.Model(input_layer, encoded)
    
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    
    # Train the autoencoder
    autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, shuffle=True, validation_data=(X_test, X_test))
    
    return autoencoder, encoder

# Example usage:
# data = load_data('data.csv')
# data_preprocessed = preprocess_data(data)
# X_train, X_test, y_train, y_test = split_data(data, 'target')
# visualize_data(data)
# linear_regression(X_train, y_train, X_test, y_test)
# logistic_regression(X_train, y_train, X_test, y_test)
# decision_tree(X_train, y_train, X_test, y_test)
# random_forest(X_train, y_train, X_test, y_test)
# support_vector_machine(X_train, y_train, X_test, y_test)
# artificial_neural_network(X_train, y_train, X_test, y_test)
# convolutional_neural_network(X_train, y_train, X_test, y_test, img_height, img_width)
# labels = kmeans_clustering(data)
# principal_components = pca_analysis(data)
# weights, means, covariances = m_step(data, e_step(data, initial_weights, initial_means, initial_covariances))
# cross_validation(LogisticRegression(), X, y)
# best_params = grid_search_tuning(X_train, y_train)
# layers = [X_train.shape[1], 64, 1]
# trained_weights, trained_biases = train_custom_nn(X_train, y_train, layers, learning_rate=0.01, epochs=1000)
# autoencoder_model, encoder_model = autoencoder(X_train, X_test, encoding_dim=32)
