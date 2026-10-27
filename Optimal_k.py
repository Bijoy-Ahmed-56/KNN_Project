import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1. Dataset Load and Split
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Loop through odd values of k from 1 to 25
k_values = range(1, 26, 2)
accuracies = []

for k in k_values:
  knn = KNeighborsClassifier(n_neighbors=k)
  knn.fit(X_train, y_train)
  y_pred = knn.predict(X_test)
  accuracies.append(accuracy_score(y_test, y_pred))

# 3. Plotting the results using Matplotlib
plt.figure(figsize=(11, 4))
plt.plot(k_values, accuracies, marker="o", linestyle="-", color="b")
plt.title("KNN Accuracy for Different Values of k")
plt.xlabel("Value of k")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# 4. Find and print the best k and accuracy
best_k = k_values[accuracies.index(max(accuracies))]
print("--- Part 3 Output ---")
print(f"Optimal value of k: {best_k}")
print(f"Maximum Accuracy: {max(accuracies)}")