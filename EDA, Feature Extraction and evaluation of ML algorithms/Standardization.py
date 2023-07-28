from scipy.io import arff
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from GBT.GBT import gbt_algorithm
from SVM.SVM import svm_algorithm
from RF.RF import rf_algorithm


def rs_dataset(dataset_path):
    dataset = arff.loadarff(dataset_path)
    data_frame = pd.DataFrame(dataset[0])
    return data_frame


def mit_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    return dataset


def rs_standard_scaling(df):
    x = df.iloc[:, :-1]
    activity_list = []
    class_list = []

    for index in range(len(x)):
        # Get each series from the data frame
        data = x.iloc[index][0]
        activity = [value for axis in data for value in axis]
        activity_list.append(activity)

    y = df.iloc[:, -1]

    for index in range(len(y)):
        class_list.append(y[index].decode('utf-8'))

    return StandardScaler().fit_transform(activity_list), LabelEncoder().fit_transform(class_list)


def mit_min_max_scaling(df):
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return MinMaxScaler().fit_transform(x), y


def conf_matrix(y_test, y_prediction, algorithm):
    labels = np.unique(y_prediction)
    results = confusion_matrix(y_test, y_prediction)
    sns.heatmap(results, xticklabels=labels, yticklabels=labels, annot=True, linewidths=0.1,
                fmt="d", cmap="YlGnBu")
    plt.title("Confusion matrix " + algorithm, fontsize = 15)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()


RS_train = rs_dataset('C:/Users/adamc/Desktop/Facultate/Licenta/An4/Invatare_automata/Tema_Invatare_automata/Racket_Sports/RacketSports_TRAIN.arff')
RS_test = rs_dataset('C:/Users/adamc/Desktop/Facultate/Licenta/An4/Invatare_automata/Tema_Invatare_automata/Racket_Sports/RacketSports_TEST.arff')
# MIT_train = mit_dataset('C:/Users/adamc/Desktop/Facultate/Licenta/An4/Invatare_automata/Tema_Invatare_automata/ECG_Heartbeat/mitbih_train.csv')
# MIT_test = mit_dataset('C:/Users/adamc/Desktop/Facultate/Licenta/An4/Invatare_automata/Tema_Invatare_automata/ECG_Heartbeat/mitbih_test.csv')

rs_x_train_scaled, rs_y_train_scaled = rs_standard_scaling(RS_train)
rs_x_test_scaled, rs_y_test_scaled = rs_standard_scaling(RS_test)
# mit_x_train_scaled, mit_y_train_scaled = mit_min_max_scaling(MIT_train)
# mit_x_test_scaled, mit_y_test_scaled = mit_min_max_scaling(MIT_test)

rs_svm_prediction = svm_algorithm(rs_x_train_scaled, rs_y_train_scaled, rs_x_test_scaled, rs_y_test_scaled)
# mit_svm_prediction = svm_algorithm(mit_x_train_scaled, mit_y_train_scaled, mit_x_test_scaled, mit_y_test_scaled)
print("\n -------------Classification Report SVM-------------\n")
print(classification_report(rs_y_train_scaled, rs_svm_prediction[:-1]))
conf_matrix(rs_y_train_scaled, rs_svm_prediction[:-1], "SVM")

rs_rf_prediction = rf_algorithm(rs_x_train_scaled, rs_y_train_scaled, rs_x_test_scaled, rs_y_test_scaled)
# mit_rf_prediction = rf_algorithm(mit_x_train_scaled, mit_y_train_scaled, mit_x_test_scaled, mit_y_test_scaled)
print("\n -------------Classification Report Random Forest-------------\n")
print(classification_report(rs_y_train_scaled, rs_rf_prediction[:-1]))
conf_matrix(rs_y_train_scaled, rs_rf_prediction[:-1], "Random Forest")

rs_gbt_prediction = gbt_algorithm(rs_x_train_scaled, rs_y_train_scaled, rs_x_test_scaled, rs_y_test_scaled)
# mit_gbt_prediction = gbt_algorithm(mit_x_train_scaled, mit_y_train_scaled, mit_x_test_scaled, mit_y_test_scaled)
print("\n -------------Classification Report Gradient Boosted Tree-------------\n")
print(classification_report(rs_y_train_scaled, rs_gbt_prediction[:-1]))
conf_matrix(rs_y_train_scaled, rs_gbt_prediction[:-1], "Gradient Boosted Tree")
