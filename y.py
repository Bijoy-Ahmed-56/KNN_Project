# 1) Initialize a KNeighborsClassifier with n_neighbors=5
knn = KNeighborsClassifier(n_neighbors=5)

# 2) Train (fit) the model on the training data
knn.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = knn.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy Score:", accuracy)
print("\nConfusion Matrix:")
print(conf_matrix)