from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. Dataset load kora ebong split kora
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=43
)

# 2. KNeighborsClassifier model toiri kora (n_neighbors=5 diye)
knn = KNeighborsClassifier(n_neighbors=4)

# 3. Model training (fit kora)
knn.fit(X_train, y_train)

# 4. Prediction kora ebong evaluation
y_pred = knn.predict(X_test)

print("--- Part 2 Output ---")
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))