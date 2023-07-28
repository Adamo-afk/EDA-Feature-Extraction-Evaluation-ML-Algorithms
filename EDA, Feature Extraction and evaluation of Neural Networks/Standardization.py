import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from GBT.GBT import gbt_algorithm
from SVM.SVM import svm_algorithm
from RF.RF import rf_algorithm


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

    return x_train, x_test, y_train, y_test


def ptb_min_max_scaling(x):
    return MinMaxScaler().fit_transform(x)


def conf_matrix(y_test, y_prediction, algorithm):
    labels = np.unique(y_prediction)
    results = confusion_matrix(y_test, y_prediction)
    sns.heatmap(results, xticklabels=labels, yticklabels=labels, annot=True, linewidths=0.1,
                fmt="d", cmap="YlGnBu")
    plt.title("Confusion matrix " + algorithm, fontsize=15)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


ptb_x_train, ptb_x_test, ptb_y_train_scaled, ptb_y_test_scaled = create_ptb_dataset()

ptb_x_train_scaled = ptb_min_max_scaling(ptb_x_train)
ptb_x_test_scaled = ptb_min_max_scaling(ptb_x_test)

# ptb_svm_prediction = svm_algorithm(ptb_x_train_scaled, ptb_y_train_scaled, ptb_x_test_scaled, ptb_y_test_scaled)
# print("\n -------------Classification Report SVM-------------\n")
# print(classification_report(ptb_y_test_scaled, ptb_svm_prediction))
# conf_matrix(ptb_y_test_scaled, ptb_svm_prediction, "SVM")

# ptb_rf_prediction = rf_algorithm(ptb_x_train_scaled, ptb_y_train_scaled, ptb_x_test_scaled, ptb_y_test_scaled)
# print("\n -------------Classification Report Random Forest-------------\n")
# print(classification_report(ptb_y_test_scaled, ptb_rf_prediction))
# conf_matrix(ptb_y_test_scaled, ptb_rf_prediction, "Random Forest")

ptb_gbt_prediction = gbt_algorithm(ptb_x_train_scaled, ptb_y_train_scaled, ptb_x_test_scaled, ptb_y_test_scaled)
print("\n -------------Classification Report Gradient Boosted Tree-------------\n")
print(classification_report(ptb_y_test_scaled, ptb_gbt_prediction))
conf_matrix(ptb_y_test_scaled, ptb_gbt_prediction, "Gradient Boosted Tree")
