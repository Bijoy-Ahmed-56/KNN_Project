# Store accuracies for each k
k_values = list(range(1, 26, 2))  # Odd values from 1 to 25
accuracies = []

# Loop over odd values of k from 1 to 25
for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    y_pred_k = knn_k.predict(X_test)
    acc = accuracy_score(y_test, y_pred_k)
    accuracies.append(acc)
    print(f"k = {k:2d}  |  Accuracy = {acc:.4f}")

# Create a line plot
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
plt.title('kNN Accuracy vs. k Value (Iris Dataset)', fontsize=14)
plt.xlabel('k (Number of Neighbors)', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.xticks(k_values)
plt.grid(True, alpha=0.3)
plt.ylim(0.85, 1.02)
plt.axhline(y=max(accuracies), color='r', linestyle='--', alpha=0.5, label=f'Best Accuracy = {max(accuracies):.4f}')
plt.legend()
plt.tight_layout()
plt.show()

# Find the best k
best_k = k_values[accuracies.index(max(accuracies))]
best_accuracy = max(accuracies)
print(f"\nOptimal k: {best_k}")
print(f"Best Accuracy: {best_accuracy:.4f}")