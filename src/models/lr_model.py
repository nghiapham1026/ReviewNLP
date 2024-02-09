from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

from utils.hyperparameter_tuning import tune_logistic_regression

# Load the vectorized features and labels
with open('../../data/processed/tfidf_features.pkl', 'rb') as f:
    X = pickle.load(f)

with open('../../data/processed/labels.pkl', 'rb') as f:
    y = pickle.load(f)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Call the tuning function to get the best model and parameters
model, best_params = tune_logistic_regression(X_train, y_train)

print(f"Best Parameters: {best_params}")

# Make predictions on the test set with the tuned model
predictions = model.predict(X_test)

# Evaluate the tuned model
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Optionally, save the trained model for later use or deployment
with open('../../models/logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model training and evaluation complete. Model saved.")
