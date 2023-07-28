import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def gbt_algorithm(x_train, y_train, x_test, y_test):
    # Create XGBoost classifier object
    gbt = xgb.XGBClassifier()

    # Define the hyperparameters to search over
    parameters = {
        'n_estimators': [100],
        'max_depth': [10],
        'learning_rate': [0.01, 0.1]
    }

    print("Performing grid search on the Gradient Boosted Trees model...")
    # Use Grid Search with Cross Validation to find the best hyperparameters
    grid_search = GridSearchCV(gbt, parameters, cv=5)
    grid_search.fit(x_train, y_train)

    print("Evaluate the Gradient Boosted Trees model...")
    # Evaluate the best model on the test set
    y_pred = grid_search.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)

    # Print the results
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best cross-validation accuracy:", grid_search.best_score_)
    print("Test set accuracy:", accuracy)

    return y_pred
