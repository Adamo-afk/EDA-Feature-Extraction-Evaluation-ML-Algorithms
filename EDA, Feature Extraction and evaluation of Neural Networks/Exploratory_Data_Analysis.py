import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


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


def ptb_class_frequency(df, dataset_title):
    # get the frequency of each class
    class_counts = df[df.columns[::-1][0]].value_counts()
    class_names = df[df.columns[::-1][0]].unique()
    class_balance = pd.DataFrame({'class': class_names, 'frequency': class_counts})

    class_balance.plot.bar(x=class_balance.columns[0], y=class_balance.columns[1], rot=0)
    plt.title(dataset_title)
    plt.show()

    # print the frequency counts
    print(class_counts)


def ptb_series_visualization(df):
    # Get unique classes
    classes = df[df.columns[::-1][0]].unique()

    # Plot a sample series for each class
    fig, axs = plt.subplots(len(classes), 1, figsize=(15, 25))

    for i, c in enumerate(classes):
        # Get the first series from the data frame
        samples = df[df[df.columns[::-1][0]] == c].iloc[0]

        # Plot accelerometer data
        axs[i].plot(list(samples))
        axs[i].set_title(f'{c} - Arrhythmia')

    plt.show()


def ptb_mean_std_plot(df):
    # Get unique classes
    classes = df[df.columns[::-1][0]].unique()

    # Create subplots for each axis of the accelerometer and gyroscope
    fig, axs = plt.subplots(len(classes), 1, figsize=(15, 10))

    for i, c in enumerate(classes):
        mean = []
        std = []

        # Get the first series from the data frame
        samples = df[df[df.columns[::-1][0]] == c]

        for index in range(len(samples)):
            mean.append(samples.iloc[index][:186].mean())
            std.append(samples.iloc[index][:186].std())

        axs[i].plot(mean, label='mean')
        axs[i].plot(std, label='std')
        axs[i].set_title(f'{c} - Arrhythmia')
        axs[i].legend()

    plt.show()


ptb_train, ptb_test = create_ptb_dataset()

ptb_class_frequency(ptb_train, 'PTB TRAIN')
ptb_class_frequency(ptb_test, 'PTB TEST')

ptb_series_visualization(ptb_train)
ptb_series_visualization(ptb_test)

ptb_mean_std_plot(ptb_train)
ptb_mean_std_plot(ptb_test)
