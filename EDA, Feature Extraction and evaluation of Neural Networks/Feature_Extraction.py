import numpy as np
from scipy.signal import find_peaks
from scipy import stats
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from GBT.GBT import gbt_algorithm
from SVM.SVM import svm_algorithm
from RF.RF import rf_algorithm


def ptb_feature_extraction(df):
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

    x_train = list(x_train)
    x_test = list(x_test)

    for index, value in enumerate(y_train):
        x_train[index] = np.append(x_train[index], value)

    for index, value in enumerate(y_test):
        x_test[index] = np.append(x_test[index], value)

    return pd.DataFrame(x_train), pd.DataFrame(x_test)


ptb_train, ptb_test = create_ptb_dataset()

ptb_x_train, ptb_y_train = ptb_feature_extraction(ptb_train)
ptb_x_test, ptb_y_test = ptb_feature_extraction(ptb_test)

ptb_x_train, ptb_x_test = feature_selection_variance_threshold(ptb_x_train, ptb_x_test)

# svm_algorithm(ptb_x_train, ptb_y_train, ptb_x_test, ptb_y_test)

# rf_algorithm(ptb_x_train, ptb_y_train, ptb_x_test, ptb_y_test)

# gbt_algorithm(ptb_x_train, ptb_y_train, ptb_x_test, ptb_y_test)
