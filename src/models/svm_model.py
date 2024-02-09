from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from hyperparameter_tuning import tune_svm  # Make sure to import the correct function
import pickle

# Load the vectorized features and labels
with open('../../data/processed/tfidf_features.pkl', 'rb') as f:
    X = pickle.load(f)

with open('../../data/processed/labels.pkl', 'rb') as f:
    y = pickle.load(f)

# Splitting the data into training and testing sets
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Further split the training data to use only half of it for training
_, X_train, _, y_train = train_test_split(X_train_full, y_train_full, test_size=0.5, random_state=42)

# Call the tuning function to get the best SVM model and parameters
model, best_params = tune_svm(X_train, y_train)

print(f"Best Parameters: {best_params}")

# The model is already trained with the best parameters during tuning
# Make predictions on the test set with the tuned model
predictions = model.predict(X_test)

# Evaluate the tuned model
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

"""
# Optionally, save the trained and tuned model for later use or deployment
with open('../../models/svm_tuned_model.pkl', 'wb') as f:
    pickle.dump(model, f)
"""

print("Model training, tuning, and evaluation complete. SVM model saved.")
