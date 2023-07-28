from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def rf_algorithm(x_train, y_train, x_test, y_test):
    # create the RandomForestClassifier model
    rf = RandomForestClassifier(random_state=42)

    # define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100],
        'max_depth': [10],
        'max_features': [0.25, 0.5]
    }

    print("Performing grid search on the Random Forest model...")
    # Perform grid search with cross validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
    grid_search.fit(x_train, y_train)

    print("Evaluate the Random Forest model...")
    # Evaluate the best model on the test set
    y_pred = grid_search.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)

    # Print the results
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best cross-validation accuracy:", grid_search.best_score_)
    print("Test set accuracy:", accuracy)

    return y_pred
