import matplotlib.pyplot as plt
from scipy.io import arff
import pandas as pd


def df_bar_plot(df, dataset_title):
    # get the frequency of each class
    class_counts = df[df.columns[::-1][0]].value_counts()
    class_names = df[df.columns[::-1][0]].unique()
    class_balance = pd.DataFrame({'class': class_names, 'frequency': class_counts})

    class_balance.plot.bar(x=class_balance.columns[0], y=class_balance.columns[1], rot=0)
    plt.title(dataset_title)
    plt.show()

    # print the frequency counts
    print(class_counts)


def rs_class_frequency(dataset_path, dataset_title):
    dataset = arff.loadarff(dataset_path)
    data_frame = pd.DataFrame(dataset[0])
    print(data_frame.columns)
    df_bar_plot(data_frame, dataset_title)
    return data_frame


def mit_class_frequency(dataset_path, dataset_title):
    dataset = pd.read_csv(dataset_path)
    print(dataset.columns)
    df_bar_plot(dataset, dataset_title)
    return dataset


def rs_series_visualization(df):
    # Get unique classes
    classes = df[df.columns[::-1][0]].unique()

    # Plot a sample series for each class
    fig, axs = plt.subplots(len(classes), 2, figsize=(15, 25))

    for i, c in enumerate(classes):
        # Get the first series from the data frame
        samples = df[df[df.columns[::-1][0]] == c].iloc[0]

        # Plot accelerometer data
        axs[i][0].plot(list(samples[0][0]), label='x')
        axs[i][0].plot(list(samples[0][1]), label='y')
        axs[i][0].plot(list(samples[0][2]), label='z')
        axs[i][0].set_title(f'{c} - Accelerometer')
        axs[i][0].legend()

        # Plot gyroscope data
        axs[i][1].plot(list(samples[0][3]), label='x')
        axs[i][1].plot(list(samples[0][4]), label='y')
        axs[i][1].plot(list(samples[0][5]), label='z')
        axs[i][1].set_title(f'{c} - Gyroscope')
        axs[i][1].legend(['x', 'y', 'z'])

    plt.show()


def mit_series_visualization(df):
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


def rs_value_distribution(df):
    # Get unique classes
    classes = df[df.columns[::-1][0]].unique()
    label_map = {0: 'X', 1: 'Y', 2: 'Z'}

    # Create subplots for each axis of the accelerometer and gyroscope
    fig, axs = plt.subplots(2, len(label_map), figsize=(15, 10))

    # Loop through each class
    for c in classes:
        # Get all examples of the current class
        samples = df[df[df.columns[::-1][0]] == c]

        # Loop through each axis of the accelerometer and gyroscope
        for i in range(len(label_map)):
            filtered_df = df.loc[df[df.columns[::-1][0]] == c]

            # Plot a histogram of the current axis for the current class
            axs[0][i].hist(list(samples.iloc[:, 0][filtered_df.index[0]][i]), bins=50, alpha=0.5, label=c)
            axs[0][i].set_title(f'{c} - Accelerometer {label_map[i]}')
            axs[0][i].legend()

            axs[1][i].hist(list(samples.iloc[:, 0][filtered_df.index[0]][i+3]), bins=50, alpha=0.5, label=c)
            axs[1][i].set_title(f'{c} - Gyroscope {label_map[i]}')
            axs[1][i].legend()

    plt.show()


def mit_mean_std_plot(df):
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


RS_train = rs_class_frequency('C:/Users/adamc/Desktop/Facultate/Licenta/An4/Invatare_automata/Tema_Invatare_automata/Racket_Sports/RacketSports_TRAIN.arff', 'Racket Sports TRAIN')
RS_test = rs_class_frequency('C:/Users/adamc/Desktop/Facultate/Licenta/An4/Invatare_automata/Tema_Invatare_automata/Racket_Sports/RacketSports_TEST.arff', 'Racket Sports TEST')
MIT_train = mit_class_frequency('C:/Users/adamc/Desktop/Facultate/Licenta/An4/Invatare_automata/Tema_Invatare_automata/ECG_Heartbeat/mitbih_train.csv', 'MIT-BIH TRAIN')
MIT_test = mit_class_frequency('C:/Users/adamc/Desktop/Facultate/Licenta/An4/Invatare_automata/Tema_Invatare_automata/ECG_Heartbeat/mitbih_test.csv', 'MIT-BIH TEST')

rs_series_visualization(RS_train)
rs_series_visualization(RS_test)

mit_series_visualization(MIT_train)
mit_series_visualization(MIT_test)

rs_value_distribution(RS_train)
rs_value_distribution(RS_test)

mit_mean_std_plot(MIT_train)
mit_mean_std_plot(MIT_test)
