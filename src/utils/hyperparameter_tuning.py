from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def tune_logistic_regression(X_train, y_train):
    # Define the parameter grid for Logistic Regression
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
        'penalty': ['l1', 'l2'],  # Norm used in the penalization
        'solver': ['liblinear']  # Solver that supports penalty='l1'
    }
    
    # Initialize the Logistic Regression model
    log_reg = LogisticRegression(max_iter=1000)
    
    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', verbose=3)
    
    # Perform the grid search on the training data
    grid_search.fit(X_train, y_train)
    
    # Return the best model
    return grid_search.best_estimator_, grid_search.best_params_


def tune_random_forest(X_train, y_train):
    # Define the parameter grid for Random Forest
    param_grid = {
        'n_estimators': [10, 50, 100, 200],  # Number of trees in the forest
        'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
        'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
        'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
    }
    
    # Initialize the Random Forest model
    rf = RandomForestClassifier(random_state=42)
    
    # Initialize the GridSearchCV object
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', verbose=3, n_jobs=-1)
    
    # Perform the grid search on the training data
    grid_search.fit(X_train, y_train)
    
    # Return the best model and the best parameters
    return grid_search.best_estimator_, grid_search.best_params_

def tune_svm(X_train, y_train):
    param_grid = {
        'C': [0.1, 1, 10],  # Regularization parameter
        'kernel': ['linear', 'rbf'],  # Kernel type
        'gamma': ['scale', 'auto']  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
    }
    
    svm = SVC(random_state=42)
    grid_search = GridSearchCV(svm, param_grid, scoring='accuracy', cv=5, verbose=3, n_jobs=-1)
    
    grid_search.fit(X_train, y_train)
    
    return grid_search.best_estimator_, grid_search.best_params_