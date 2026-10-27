from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 1. Dataset load kora
iris = load_iris()
X, y = iris.data, iris.target

# 2. Data split kora (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=43
)

print("Part 1 Success: Data loaded and split successfully!")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)