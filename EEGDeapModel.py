import os

import mne
import pandas as pd
import scipy
import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.integrate import simps
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC


def preprocessing_data(file="C:/Users/alash/Desktop/4 курс/Diplom/program/datasets/DEAP/data_original/s01.bdf"):
    # set sampling frequency to 128 Hz
    sampling_freq = 128
    # create MNE info object
    info = mne.create_info(32, sfreq=sampling_freq)

    raw = mne.io.read_raw_bdf(file, preload=True)
    raw.plot()

    # print information about the raw data
    print(raw.info)

    #  bandpass frequency filter
    raw_data = raw.filter(l_freq=4, h_freq=45, fir_design='firwin', l_trans_bandwidth='auto', filter_length='auto')

    ica = mne.preprocessing.ICA(n_components=20, random_state=0)
    ica.fit(raw)
    raw_data = ica.apply(raw)

    # plot data again after removing bad channels and interpolating
    raw_data.compute_psd(fmax=50).plot(picks="data", exclude="bads")
    raw_data.plot(block=True)

    data = scipy.io.loadmat('C:/Users/alash/Desktop/4 курс/Diplom/program/datasets/SEED_EEG/SEED_EEG/Preprocessed_EEG/1_20131027.mat')
    print(data)

    return raw


def emotional_labeling(arousal, valence):
    emotions_set = []
    arousal = arousal - 4.5
    valence = valence - 4.5
    for index in range(arousal.size):
        if arousal[index] > 0 and valence[index] > 0:
            # happy
            emotions_set.append(1)
        elif arousal[index] > 0 > valence[index]:
            # angry
            emotions_set.append(2)
        elif arousal[index] < 0 < valence[index]:
            # calm
            emotions_set.append(3)
        elif arousal[index] < 0 and valence[index] < 0:
            # sad
            emotions_set.append(4)
        else:
            emotions_set.append(0)
    return emotions_set


def read_preprocessed_data():
    print("Выделение первоначальных датафреймов для eeg данных и признаков:\n")
    directory = "C:/Users/alash/Desktop/4 курс/Diplom/program/datasets/DEAP/data_preprocessed_matlab/"
    data = []
    labels = []
    files = os.listdir(directory)
    for file in files:
        current_file = spio.loadmat(directory + file)

        keys = [key for key, values in current_file.items() if
                key != '__header__' and key != '__version__' and key != '__globals__']

        labels.append(current_file[keys[0]])
        data.append(current_file[keys[1]])

    labels = np.array(labels)
    data = np.array(data)

    print(files)
    print(labels.shape)
    print(data.shape)

    labels = labels.reshape(1280, 4)
    data = data.reshape(1280, 40, 8064)

    print(labels.shape)
    print(data.shape)

    return labels, data


def feature_extraction(labels, data):
    print("Выделение признаков из eeg данных на основе диапазона мощности каждого из сигналов:\n")
    eeg_data = data[:, :32, :]
    labels = labels[:, :2]
    print(labels.shape)
    print(eeg_data.shape)

    eeg_band = []
    for i in range(len(eeg_data)):
        for j in range(len(eeg_data[0])):
            eeg_band.append(get_band_power(i, j, "theta", eeg_data))
            eeg_band.append(get_band_power(i, j, "alpha", eeg_data))
            eeg_band.append(get_band_power(i, j, "beta", eeg_data))
            eeg_band.append(get_band_power(i, j, "gamma", eeg_data))

    eeg_band = (np.array(eeg_band))
    eeg_band = eeg_band.reshape((1280, 128))
    print(eeg_band.shape)

    np.save("datasets2/eeg_band.npy", eeg_band)

    return eeg_band


def create_labels(eeg_band_data_):
    print("Маркировка тренировочного набора arousal, valence соответствующими эмоциями:\n")
    data_with_labels = pd.DataFrame({'Valence': eeg_band_data_[:, 0] - 4.5, 'Arousal': eeg_band_data_[:, 1] - 4.5,
                                     'Emotion': emotional_labeling(eeg_band_data_[:, 0], eeg_band_data_[:, 1])})
    data_with_labels.info()
    data_with_labels.describe()
    df_label_ratings = pd.DataFrame({'Valence': eeg_band_data_[:, 0], 'Arousal': eeg_band_data_[:, 1]})
    # Построим график первых 40 строк данных
    df_label_ratings.iloc[0:40].plot(style=['o', 'rx'])
    np.save("eeg_labels", data_with_labels)

    create_circumplex_model(data_with_labels)

    return data_with_labels


def create_circumplex_model(dataframe):
    # Установите общей фигуры и осей
    fig, ax = plt.subplots(figsize=(12, 8))

    # Установка лимита для осей
    ax.set_xlim(-4.6, 4.6)
    ax.set_ylim(-4.6, 4.6)
    print(dataframe)
    # Установка точек и параметров для соответствующих эмоций
    for index, row in dataframe.iterrows():
        valence = row['Valence']
        arousal = row['Arousal']

        # Соотношение значений валентности и возбуждения с координатами в модели
        x = valence
        y = arousal
        color = 'white'
        marker = "x"
        label = "neutral"
        if y > 0 and x > 0:
            color = 'red'
            marker = "*"
            label = "happy"
        elif y > 0 > x:
            color = 'black'
            marker = "v"
            label = "angry"
        elif y < 0 < x:
            color = 'blue'
            marker = "D"
            label = "calm"
        elif y < 0 and x < 0:
            color = 'green'
            marker = "."
            label = "sad"

        ax.scatter(x, y, s=11, color=color, label=label, alpha=0.5, marker=marker)

    ax.set_xlabel('Valence')
    ax.set_ylabel('Arousal')
    ax.grid(True)

    legend_elements = [
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=8, label='Happy'),
        plt.Line2D([0], [0], marker='v', color='w', markerfacecolor='black', markersize=8, label='Angry'),
        plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='blue', markersize=8, label='Calm'),
        plt.Line2D([0], [0], marker='.', color='w', markerfacecolor='green', markersize=8, label='Sad')
    ]
    ax.legend(handles=legend_elements, loc="lower left")
    plt.show()


def svm_classifier(data, labels):
    print("Классификации методом опорных векторов")
    """
    Функция для обучения классификатора SVM на наборе данных и выполнения классификации.

    Аргументы:
    - data: np.array, набор данных размером (1280, 160) с EEG измерениями.
    - labels: np.array, метки классов размером (1280, 3) в формате [arousal (float), valence (float), emotion (string)].

    Возвращает:
    - accuracy: float, точность классификации на тестовых данных.
    - report: str, отчет о классификации (precision, recall, f1-score) на тестовых данных.
    """

    # Извлечение отдельных столбцов меток классов
    emotion = labels[:, 2]

    # Разделение данных на тренировочную и тестовую выборки
    # data - матрица объектов признаков и emotion - вектор ответов
    X_train, X_test, y_train, y_test = train_test_split(data, emotion, test_size=0.3, random_state=42)

    # Создание и обучение классификатора SVM
    classifier = SVC(decision_function_shape='ovr')
    classifier.fit(X_train, y_train)
    # Прогнозирование меток классов на тестовых данных
    predicted = classifier.predict(X_test)
    # Оценка точности классификации
    accuracy = accuracy_score(predicted, y_test)
    # Генерация отчета о классификации
    report = classification_report(y_test, predicted, zero_division=1)
    print("Точность классификации 4х эмоций: " + str(accuracy))
    print(report)

    return accuracy


def random_forest_classifier(data, labels):
    print("Классификации методом random_forest")
    """
    Функция для обучения классификатора Random Forest на наборе данных и выполнения классификации.

    Аргументы:
    - data: np.array, набор данных размером (1280, 160) с EEG измерениями.
    - labels: np.array, метки классов размером (1280, 3) в формате [arousal (float), valence (float), emotion (string)].

    Возвращает:
    - accuracy: float, точность классификации на тестовых данных.
    - report: str, отчет о классификации (precision, recall, f1-score) на тестовых данных.
    """

    # Извлечение отдельных столбцов меток классов
    emotion = labels[:, 2]

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(data, emotion, test_size=0.3, random_state=42)

    # Создание и обучение классификатора Random Forest
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    # Прогнозирование меток классов на тестовых данных
    y_pred = classifier.predict(X_test)

    # Оценка точности классификации
    accuracy = accuracy_score(y_test, y_pred)

    # Генерация отчета о классификации
    report = classification_report(y_test, y_pred, zero_division=1)

    print("Точность классификации 4х эмоций: " + str(accuracy))
    print(report)

    return accuracy


def classification_knn(eeg_band, labels):
    print("Классификации методом ближайших соседей")
    """
    Функция для обучения классификатора  на наборе данных и выполнения классификации.

    Аргументы:
    - data: np.array, набор данных размером (1280, 160) с EEG измерениями.
    - labels: np.array, метки классов размером (1280, 3) в формате [arousal (float), valence (float), emotion (string)].

    Возвращает:
    - accuracy: float, точность классификации на тестовых данных.
    - report: str, отчет о классификации (precision, recall, f1-score) на тестовых данных.
    """

    labels_valence = []
    labels_arousal = []
    labels_emotion = []
    for la in labels:
        if la[0] > 0:
            labels_valence.append(1)
        else:
            labels_valence.append(0)
        if la[1] > 0:
            labels_arousal.append(1)
        else:
            labels_arousal.append(0)
        if la[2] == "happy":
            labels_emotion.append(0)
        elif la[2] == "angry":
            labels_emotion.append(1)
        elif la[2] == "sad":
            labels_emotion.append(2)
        elif la[2] == "calm":
            labels_emotion.append(3)

    X = eeg_band
    # Измерение обновления
    poly = preprocessing.PolynomialFeatures(degree=2)
    X = poly.fit_transform(X)

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    X = preprocessing.normalize(X, norm='l1')

    emotion = labels[:, 2]
    X_train, X_test, y_train, y_test = train_test_split(X, emotion, test_size=0.3, random_state=42)

    knn = KNeighborsClassifier(n_neighbors=4)
    knn.fit(X_train, y_train)
    train_score = knn.score(X_train, y_train)
    test_score = knn.score(X_test, y_test)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print("Точность：", accuracy)
    report = classification_report(y_test, y_pred, zero_division=1)
    print(report)
    return accuracy


def bandpower(data, sf, band):
    sns.set(font_scale=1.2)

    band = np.asarray(band)
    # Определить нижний и верхний пределы дельты
    low, high = band
    #  График сигнала первого пробного и последнего каналов
    # plt.plot(data, lw=1.5, color='k')
    # plt.xlabel('Time')
    # plt.ylabel('Voltage')
    # sns.despine()
    # ----------------------------------------------------
    # Определение длины окна (4 секунды)
    window = (2 / low) * sf
    frequency, psd = signal.welch(data, sf, nperseg=window)

    # График спектра мощности
    # sns.set(font_scale=1.2, style='white')
    # plt.figure(figsize=(8, 4))
    # plt.plot(frequency, psd, color='k', lw=2)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power spectral density (uV^2 / Hz)')
    # plt.ylim([0, psd.max() * 1.1])
    # plt.title("Welch's periodogram")
    # plt.xlim([0, frequency.max()])
    # sns.despine()
    # -------------------------
    # Частотное разрешение
    freq_res = frequency[1] - frequency[0]
    # Пересекающиеся значения в частотном векторе
    idx_band = np.logical_and(frequency >= low, frequency <= high)

    #  Прежде чем вычислять среднюю мощность дельта-диапазона, нужно найти диапазоны частот, которые пересекают дельта-диапазон частот.
    # График спектральной плотности мощности и заполним тета-область
    # plt.figure(figsize=(8, 4))
    # plt.plot(frequency, psd, lw=2, color='k')
    # plt.fill_between(frequency, psd, where=idx_band, color='skyblue')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Power spectral density (uV^2 / Hz)')
    # plt.xlim([0, 50])
    # plt.ylim([0, psd.max() * 1.1])
    # plt.title("Welch's periodogram")
    # sns.despine()
    # ------------------------------------------------------

    band_power = simps(psd[idx_band], dx=freq_res)
    return band_power


def get_band_power(people, channel, band, eeg_data):
    bd = (0, 0)
    if band == "theta":
        bd = (4, 8)
    elif band == "alpha":
        bd = (8, 12)
    elif band == "beta":
        bd = (12, 30)
    elif band == "gamma":
        bd = (30, 64)
    return bandpower(eeg_data[people, channel], 128., bd)


def accuracy_compare_plot(result_rf, result_svm, result_knn):
    accuracy = [result_rf, result_svm, result_knn]

    model_name = ["RF", "SVM", "KNN"]

    plt.title('Model Score', fontsize=16)
    plt.xlabel('model', fontsize=14)
    plt.ylabel('score', fontsize=14)
    plt.grid(linestyle=':', axis='y')
    x = np.arange(3)
    a = plt.bar(x, accuracy, 0.3, color='orangered', label='test', align='center')

    for i in a:
        h = i.get_height()
        plt.text(i.get_x() + i.get_width() / 2, h, '%.3f' % h, ha='center', va='bottom')
    plt.xticks(x, model_name, rotation=75)
    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    # 1 Предварительная обработка данных
    preprocessing_data()
    # 2 Выделение первоначальных датафреймов для eeg данных и признаков
    # labels_for_feature, data_for_feature = read_preprocessed_data()
    # 3 Выделение признаков из eeg данных на основе диапазона мощности каждого из сигналов
    #eeg_band_data = feature_extraction(labels_for_feature, data_for_feature)
    # 4 Маркировка тренировочного набора arousal, valence соответствующими эмоциями
    # 1 - happy, 2 - angry, 3 - calm, 4 - sad.
    # labels_for_classification = create_labels(labels_for_feature)
    # 5 Классификация различными методам машинного обучения
    # данные берутся из заранее сохраненных результатов предыдущих методов для ускорения работ
    # labels_for_classification = 'datasets2/eeg_labels.npy'
    # eeg_band_data = 'datasets2/eeg_band.npy'
    #
    # # classification with random_forest classificator
    # result_rf = random_forest_classifier(np.load(eeg_band_data, allow_pickle=True),
    #                                       np.load(labels_for_classification, allow_pickle=True))
    # # classification with svm classificator
    # result_svm = svm_classifier(np.load(eeg_band_data, allow_pickle=True),
    #                              np.load(labels_for_classification, allow_pickle=True))
    # # classification with knn classificator
    # result_knn = classification_knn(np.load(eeg_band_data, allow_pickle=True),
    #                                  np.load(labels_for_classification, allow_pickle=True))
    #
    # accuracy_compare_plot(result_rf, result_svm, result_knn)
