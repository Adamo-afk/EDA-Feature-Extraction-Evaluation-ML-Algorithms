import numpy as np
from scipy.io import arff
from scipy.signal import find_peaks
from scipy import stats
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from GBT.GBT import gbt_algorithm
from SVM.SVM import svm_algorithm
from RF.RF import rf_algorithm


def rs_feature_extraction(df):
    x = df.iloc[:, :-1]
    class_list = []
    column_lists = []
    prefixes = ['x_acc_', 'y_acc_', 'z_acc_', 'x_gyro_', 'y_gyro_', 'z_gyro_']
    suffixes = ['mean', 'std', 'aad', 'min', 'max', 'median', 'mad', 'IQR', 'neg_count', 'pos_count',
                'above_mean', 'peak_count', 'skewness', 'kurtosis', 'energy']

    for prefix in prefixes:
        for suffix in suffixes:
            column_lists.append(prefix + suffix)

    column_lists.append('avg_result_accl_acc')
    column_lists.append('avg_result_accl_gyro')
    column_lists.append('sma_acc')
    column_lists.append('sma_gyro')

    x_train = pd.DataFrame(columns=column_lists)

    for index in range(len(x)):
        # Get each series from the data frame
        data = x.iloc[index][0]
        column_list_counter = 0
        for axis in range(len(data)):
            # mean
            x_train.loc[index, column_lists[column_list_counter]] = np.mean(list(data[axis]))
            column_list_counter += 1
            # std dev
            x_train.loc[index, column_lists[column_list_counter]] = np.std(list(data[axis]))
            column_list_counter += 1
            # avg absolute diff
            x_train.loc[index, column_lists[column_list_counter]] = np.mean(
                np.absolute(list(data[axis]) - np.mean(list(data[axis]))))
            column_list_counter += 1
            # min
            x_train.loc[index, column_lists[column_list_counter]] = np.min(list(data[axis]))
            column_list_counter += 1
            # max
            x_train.loc[index, column_lists[column_list_counter]] = np.max(list(data[axis]))
            column_list_counter += 1
            # median
            x_train.loc[index, column_lists[column_list_counter]] = np.median(list(data[axis]))
            column_list_counter += 1
            # median abs dev
            x_train.loc[index, column_lists[column_list_counter]] = np.median(
                np.absolute(list(data[axis]) - np.median(list(data[axis]))))
            column_list_counter += 1
            # interquartile range
            x_train.loc[index, column_lists[column_list_counter]] = np.percentile(
                list(data[axis]), 75) - np.percentile(list(data[axis]), 25)
            column_list_counter += 1
            # negative count
            x_train.loc[index, column_lists[column_list_counter]] = np.sum(np.array(list(data[axis])) < 0)
            column_list_counter += 1
            # positive count
            x_train.loc[index, column_lists[column_list_counter]] = np.sum(np.array(list(data[axis])) > 0)
            column_list_counter += 1
            # values above mean
            x_train.loc[index, column_lists[column_list_counter]] = np.sum(
                np.array(list(data[axis])) > np.mean(list(data[axis])))
            column_list_counter += 1
            # number of peaks
            x_train.loc[index, column_lists[column_list_counter]] = len(find_peaks(list(data[axis]))[0])
            column_list_counter += 1
            # skewness
            x_train.loc[index, column_lists[column_list_counter]] = stats.skew(np.array(list(data[axis])))
            column_list_counter += 1
            # kurtosis
            x_train.loc[index, column_lists[column_list_counter]] = stats.kurtosis(np.array(list(data[axis])))
            column_list_counter += 1
            # energy
            x_train.loc[index, column_lists[column_list_counter]] = np.sum((np.array(list(data[axis]))**2)/100)
            column_list_counter += 1

        # avg resultant
        x_train.loc[index, 'avg_result_accl_acc'] = np.mean(
            (np.array(list(data[0])) ** 2 + np.array(list(data[1])) ** 2 + np.array(list(data[2])) ** 2) ** 0.5)
        column_list_counter += 1
        x_train.loc[index, 'avg_result_accl_gyro'] = np.mean(
            (np.array(list(data[3])) ** 2 + np.array(list(data[4])) ** 2 + np.array(list(data[5])) ** 2) ** 0.5)
        column_list_counter += 1

        # signal magnitude area
        x_train.loc[index, 'sma_acc'] = np.sum(abs(np.array(list(data[0])) / 100)) +\
                                        np.sum(abs(np.array(list(data[1])) / 100)) +\
                                        np.sum(abs(np.array(list(data[0])) / 100))
        column_list_counter += 1
        x_train.loc[index, 'sma_gyro'] = np.sum(abs(np.array(list(data[3])) / 100)) +\
                                         np.sum(abs(np.array(list(data[4])) / 100)) +\
                                         np.sum(abs(np.array(list(data[5])) / 100))
        column_list_counter += 1

    y = df.iloc[:, -1]

    for index in range(len(y)):
        class_list.append(y[index].decode('utf-8'))

    return x_train, LabelEncoder().fit_transform(class_list)


def mit_feature_extraction(df):
    x = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    column_lists = ['mean', 'std', 'aad', 'min', 'max', 'median', 'mad', 'IQR', 'neg_count', 'pos_count',
                    'above_mean', 'peak_count', 'skewness', 'kurtosis', 'energy', 'avg_result_accl', 'sma']

    x_train = pd.DataFrame(columns=column_lists)

    for index in range(len(x)):
        # Get each series from the data frame
        data = np.array(x.iloc[index])
        column_list_counter = 0
        # mean
        x_train.loc[index, column_lists[column_list_counter]] = np.mean(list(data))
        column_list_counter += 1
        # std dev
        x_train.loc[index, column_lists[column_list_counter]] = np.std(list(data))
        column_list_counter += 1
        # avg absolute diff
        x_train.loc[index, column_lists[column_list_counter]] = np.mean(
            np.absolute(list(data) - np.mean(list(data))))
        column_list_counter += 1
        # min
        x_train.loc[index, column_lists[column_list_counter]] = np.min(list(data))
        column_list_counter += 1
        # max
        x_train.loc[index, column_lists[column_list_counter]] = np.max(list(data))
        column_list_counter += 1
        # median
        x_train.loc[index, column_lists[column_list_counter]] = np.median(list(data))
        column_list_counter += 1
        # median abs dev
        x_train.loc[index, column_lists[column_list_counter]] = np.median(
            np.absolute(list(data) - np.median(list(data))))
        column_list_counter += 1
        # interquartile range
        x_train.loc[index, column_lists[column_list_counter]] = np.percentile(list(data), 75) - np.percentile(
            list(data), 25)
        column_list_counter += 1
        # negative count
        x_train.loc[index, column_lists[column_list_counter]] = np.sum(np.array(list(data)) < 0)
        column_list_counter += 1
        # positive count
        x_train.loc[index, column_lists[column_list_counter]] = np.sum(np.array(list(data)) > 0)
        column_list_counter += 1
        # values above mean
        x_train.loc[index, column_lists[column_list_counter]] = np.sum(
            np.array(list(data)) > np.mean(list(data)))
        column_list_counter += 1
        # number of peaks
        x_train.loc[index, column_lists[column_list_counter]] = len(find_peaks(list(data))[0])
        column_list_counter += 1
        # skewness
        x_train.loc[index, column_lists[column_list_counter]] = stats.skew(np.array(list(data)))
        column_list_counter += 1
        # kurtosis
        x_train.loc[index, column_lists[column_list_counter]] = stats.kurtosis(np.array(list(data)))
        column_list_counter += 1
        # energy
        x_train.loc[index, column_lists[column_list_counter]] = np.sum((np.array(list(data)) ** 2) / 100)
        column_list_counter += 1
        # avg resultant
        x_train.loc[index, column_lists[column_list_counter]] = np.mean(
            (np.array(list(data)) ** 2) ** 0.5)
        column_list_counter += 1
        # signal magnitude area
        x_train.loc[index, column_lists[column_list_counter]] = np.sum(abs(np.array(list(data)) / 100))

    return x_train, y


def feature_selection_variance_threshold(x_train, x_test):
    selector = VarianceThreshold(threshold=0.5)
    x_train_selection = selector.fit_transform(x_train)
    x_test_selection = selector.transform(x_test)
    return x_train_selection, x_test_selection


def rs_dataset(dataset_path):
    dataset = arff.loadarff(dataset_path)
    data_frame = pd.DataFrame(dataset[0])
    return data_frame


def mit_dataset(dataset_path):
    dataset = pd.read_csv(dataset_path)
    return dataset


RS_train = rs_dataset('C:/Users/adamc/Desktop/Facultate/Licenta/An4/Invatare_automata/Tema_Invatare_automata/Racket_Sports/RacketSports_TRAIN.arff')
RS_test = rs_dataset('C:/Users/adamc/Desktop/Facultate/Licenta/An4/Invatare_automata/Tema_Invatare_automata/Racket_Sports/RacketSports_TEST.arff')
# MIT_train = mit_dataset('C:/Users/adamc/Desktop/Facultate/Licenta/An4/Invatare_automata/Tema_Invatare_automata/ECG_Heartbeat/mitbih_train.csv')
# MIT_test = mit_dataset('C:/Users/adamc/Desktop/Facultate/Licenta/An4/Invatare_automata/Tema_Invatare_automata/ECG_Heartbeat/mitbih_test.csv')

rs_x_train, rs_y_train = rs_feature_extraction(RS_train)
rs_x_test, rs_y_test = rs_feature_extraction(RS_test)
# mit_x_train, mit_y_train = mit_feature_extraction(MIT_train)
# mit_x_test, mit_y_test = mit_feature_extraction(MIT_test)

rs_x_train, rs_x_test = feature_selection_variance_threshold(rs_x_train, rs_x_test)
# mit_x_train, mit_x_test = feature_selection_variance_threshold(mit_x_train, mit_x_test)

# svm_algorithm(rs_x_train, rs_y_train, rs_x_test, rs_y_test)
# svm_algorithm(mit_x_train, mit_y_train, mit_x_test, mit_y_test)

# rf_algorithm(rs_x_train, rs_y_train, rs_x_test, rs_y_test)
# rf_algorithm(mit_x_train, mit_y_train, mit_x_test, mit_y_test)

# gbt_algorithm(rs_x_train, rs_y_train, rs_x_test, rs_y_test)
# gbt_algorithm(mit_x_train, mit_y_train, mit_x_test, mit_y_test)
