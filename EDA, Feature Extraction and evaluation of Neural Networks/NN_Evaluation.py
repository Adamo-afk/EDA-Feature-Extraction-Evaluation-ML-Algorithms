import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from MLP.MLP import mlp_algorithm
from LSTM.LSTM import lstm_algorithm
from CNN.CNN import cnn_algorithm


def create_ptb_dataset():
    # Load PTB normal data
    normal_data = pd.read_csv('C:/Users/adamc/Desktop/Facultate/Licenta/An4/Invatare_automata/Tema_Invatare_automata/ECG_Heartbeat/ptbdb_normal.csv')

    # Load PTB abnormal data
    abnormal_data = pd.read_csv('C:/Users/adamc/Desktop/Facultate/Licenta/An4/Invatare_automata/Tema_Invatare_automata/ECG_Heartbeat/ptbdb_abnormal.csv')

    # Replace missing values with the mean of each column
    normal_data = normal_data.fillna(value=0)
    abnormal_data = abnormal_data.fillna(value=0)

    data = np.concatenate((normal_data, abnormal_data), axis=0)
    data = pd.DataFrame(data)

    # Shuffle the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # Extract x and y from the data
    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Handle missing values
    x = SimpleImputer().fit_transform(x)

    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    return x_train, x_test, y_train.values, y_test.values


def create_mit_dataset():
    train = pd.read_csv('C:/Users/adamc/Desktop/Facultate/Licenta/An4/Invatare_automata/Tema_Invatare_automata/ECG_Heartbeat/mitbih_train.csv')
    x_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1]

    test = pd.read_csv('C:/Users/adamc/Desktop/Facultate/Licenta/An4/Invatare_automata/Tema_Invatare_automata/ECG_Heartbeat/mitbih_test.csv')
    x_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]

    return x_train.values, x_test.values, y_train.values, y_test.values


def plot_training(loss, acc, epochs, title):
    # Construct a plot that plots the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epochs), loss, label="train_loss")
    plt.plot(np.arange(0, epochs), acc, label="train_acc")
    plt.title(title + " Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()


def conf_matrix(y_test, y_prediction, algorithm):
    labels = np.unique(y_prediction)
    results = confusion_matrix(y_test, y_prediction)
    sns.heatmap(results, xticklabels=labels, yticklabels=labels, annot=True, linewidths=0.1,
                fmt="d", cmap="YlGnBu")
    plt.title("Confusion matrix " + algorithm, fontsize=15)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


# ptb_x_train, ptb_x_test, ptb_y_train_scaled, ptb_y_test_scaled = create_ptb_dataset()
# ptb_mlp_prediction = mlp_algorithm(ptb_x_train, ptb_y_train_scaled, ptb_x_test, ptb_y_test_scaled)
# ptb_lstm_prediction, ptb_lstm_loss, ptb_lstm_acc, ptb_lstm_epochs = lstm_algorithm(ptb_x_train, ptb_y_train_scaled, ptb_x_test, ptb_y_test_scaled)
# ptb_cnn_prediction, ptb_cnn_loss, ptb_cnn_acc, ptb_cnn_epochs = cnn_algorithm(ptb_x_train, ptb_y_train_scaled, ptb_x_test, ptb_y_test_scaled)

mit_x_train, mit_x_test, mit_y_train_scaled, mit_y_test_scaled = create_mit_dataset()
# mit_mlp_prediction = mlp_algorithm(mit_x_train, mit_y_train_scaled, mit_x_test, mit_y_test_scaled)
# mit_lstm_prediction, mit_lstm_loss, mit_lstm_acc, mit_lstm_epochs = lstm_algorithm(mit_x_train, mit_y_train_scaled, mit_x_test, mit_y_test_scaled)
mit_cnn_prediction, mit_cnn_loss, mit_cnn_acc, mit_cnn_epochs = cnn_algorithm(mit_x_train, mit_y_train_scaled, mit_x_test, mit_y_test_scaled)

# print("\n -------------Classification Report MLP-------------\n")
# print(classification_report(ptb_y_test_scaled, ptb_mlp_prediction))
# conf_matrix(ptb_y_test_scaled, ptb_mlp_prediction, "MLP")
# print(classification_report(mit_y_test_scaled, mit_mlp_prediction))
# conf_matrix(mit_y_test_scaled, mit_mlp_prediction, "MLP")

# print("\n -------------Classification Report LSTM-------------\n")
# print(classification_report(ptb_y_test_scaled, ptb_lstm_prediction))
# conf_matrix(ptb_y_test_scaled, ptb_lstm_prediction, "LSTM")
# plot_training(ptb_lstm_loss, ptb_lstm_acc, ptb_lstm_epochs, "LSTM")
# print(classification_report(mit_y_test_scaled, mit_lstm_prediction))
# conf_matrix(mit_y_test_scaled, mit_lstm_prediction, "LSTM")
# plot_training(mit_lstm_loss, mit_lstm_acc, mit_lstm_epochs, "LSTM")

print("\n -------------Classification Report CNN-------------\n")
# print(classification_report(ptb_y_test_scaled, ptb_cnn_prediction))
# conf_matrix(ptb_y_test_scaled, ptb_cnn_prediction, "CNN")
# plot_training(ptb_cnn_loss, ptb_cnn_acc, ptb_cnn_epochs, "CNN")
print(classification_report(mit_y_test_scaled, mit_cnn_prediction))
conf_matrix(mit_y_test_scaled, mit_cnn_prediction, "CNN")
plot_training(mit_cnn_loss, mit_cnn_acc, mit_cnn_epochs, "CNN")
