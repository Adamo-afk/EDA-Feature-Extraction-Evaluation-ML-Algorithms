from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


def mlp_algorithm(x_train, y_train, x_test, y_test):
    # Create MLP classifier object
    mlp_model = MLPClassifier(max_iter=500)

    # Define the hyperparameters to search over
    parameters = {
        'hidden_layer_sizes': [(3, 2), (4, 2)]
    }

    # Use Grid Search with Cross Validation to find the best hyperparameters
    grid_search = GridSearchCV(mlp_model, parameters, cv=5)
    grid_search.fit(x_train, y_train)

    # Print the best hyperparameters found
    print("Best hyperparameters:", grid_search.best_params_)

    # Evaluate the model on the test data
    accuracy = grid_search.score(x_test, y_test)

    # Print the accuracy of the model on the test data
    print("Accuracy on test data: {:.2f}%".format(accuracy*100))

    # Print classification report
    y_pred = grid_search.predict(x_test)

    return y_pred
