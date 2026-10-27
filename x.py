# Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# 1) Load the Iris dataset
iris = load_iris()
X = iris.data    # Features (sepal length, sepal width, petal length, petal width)
y = iris.target  # Target labels (0: setosa, 1: versicolor, 2: virginica)

print("Dataset shape:", X.shape)
print("Number of classes:", len(iris.target_names))
print("Class names:", iris.target_names)

# 2) Split the data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining set size:", X_train.shape)
print("Testing set size:", X_test.shape)