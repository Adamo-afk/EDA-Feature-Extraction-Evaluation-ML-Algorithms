from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score


def svm_algorithm(x_train, y_train, x_test, y_test):
    # Define the SVM classifier
    svm = SVC(random_state=42)

    # Define the hyperparameter grid
    param_grid = {
        'C': [0.1],
        'kernel': ['linear', 'poly']
    }

    print("Performing grid search on the SVM model...")
    # Perform grid search with cross validation
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    print("Evaluate the SVM model...")
    # Evaluate the best model on the test set
    y_pred = grid_search.predict(x_test)
    accuracy = accuracy_score(y_pred, y_test)

    # Print the results
    print("Best hyperparameters:", grid_search.best_params_)
    print("Best cross-validation accuracy:", grid_search.best_score_)
    print("Test set accuracy:", accuracy)

    return y_pred
