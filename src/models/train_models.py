import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load the vectorized features and labels
with open('../../data/processed/tfidf_features.pkl', 'rb') as f:
    X = pickle.load(f)

with open('../../data/processed/labels.pkl', 'rb') as f:
    y = pickle.load(f)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=1000)  # Adjust max_iter if necessary
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# Optionally, save the trained model for later use or deployment
with open('../../models/logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model training and evaluation complete. Model saved.")
