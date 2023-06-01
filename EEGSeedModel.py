# offline recognize emotion
# using libraries for project
import mne
import pandas as pd
import numpy as np
import os

import scipy
import scipy.io as spio
import pywt

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.preprocessing import normalize
from sklearn.svm import SVC

directory = "C:/Users/alash/Desktop/4 курс/Diplom/program/EEG-Offline-Recognize-Emotions/datasets/"
path_for_file = "C:/Users/alash/Desktop/4 курс/Diplom/program/datasets/SEED_Multimodal/SEED_Multimodal/Chinese/01" \
                "-EEG-raw/1_1.cnt"

# Данная программа будет состоять из нескольких концептуальных частей:
# 1. feature_extraction/selection - выделяем только те данные, которые несут полезную информацию
# 2. principal component analysis
# 2. classification - классификация сигналов в соответствии с их свойствами,
# разнесение сигналов по соответствующим им эмоциям
# 3. visualisation - вставить красивую визуализацию
# Конечная цель - рабочая система, которая будет по загруженным в неё данным определять человеческую эмоцию


def preprocessing_data(file="C:/Users/alash/Desktop/4 курс/Diplom/program/datasets/SEED_Multimodal/SEED_Multimodal/Chinese/01-EEG-raw/1_1.cnt"):
    # set sampling frequency to 200 Hz
    sampling_freq = 200
    # create MNE info object
    info = mne.create_info(32, sfreq=sampling_freq)

    raw = mne.io.read_raw_cnt(file, preload=True)
    raw.plot()

    # print information about the raw data
    print(raw.info)

    #  bandpass frequency filter
    raw_data = raw.filter(l_freq=1, h_freq=70, fir_design='firwin', l_trans_bandwidth='auto', filter_length='auto')

    # ica = mne.preprocessing.ICA(n_components=20, random_state=0)
    # ica.fit(raw)
    # raw_data = ica.apply(raw)

    # plot data again after removing bad channels and interpolating
    raw_data.compute_psd(fmax=50).plot(picks="data", exclude="bads")
    raw_data.plot(block=True)

    data = scipy.io.loadmat('C:/Users/alash/Desktop/4 курс/Diplom/program/datasets/SEED_EEG/SEED_EEG/Preprocessed_EEG/1_20131027.mat')
    print(data)

    return raw


def feature_extraction():
    participant_trial = []
    features_table = pd.DataFrame(columns=range(620))
    files = os.listdir("C:/Users/alash/Desktop/4 курс/Diplom/program/datasets/SEED_EEG/SEED_EEG/Preprocessed_EEG/")
    for file in files:
        mat_file = spio.loadmat(
            "C:/Users/alash/Desktop/4 курс/Diplom/program/datasets/SEED_EEG/SEED_EEG/Preprocessed_EEG/" + file)
        keys = [key for key, values in mat_file.items() if
                key != '__header__' and key != '__version__' and key != '__globals__']
        for data_file in keys:
            data_df = pd.DataFrame(mat_file[data_file])

            m = data_df.shape[0]
            n = data_df.shape[1]
            # Извлечение функций модуля
            entropy = []
            energy = []
            for channel in data_df.iterrows():  # Итерация по 62 каналам
                dwt_bands = []
                data = channel[1]
                int_ent = []
                int_eng = []
                for band in range(5):
                    (data, coeff_d) = pywt.dwt(data, "db6")
                    dwt_bands.append(coeff_d)

                for band in range(len(dwt_bands)):  # DWT_bands = 23504, 11755
                    int_ent.append(calc_shannon_entropy(dwt_bands[len(dwt_bands) - band - 1]))
                    int_eng.append(calc_wavelet_energy(dwt_bands[len(dwt_bands) - band - 1]))

                entropy.append(int_ent)
                energy.append(int_eng)

            unroll_entropy = []
            unroll_energy = []
            # Преобразование 2D-массива в 1D-вектор функций, а затем объединение двух одномерных массивов.
            for i in range(len(entropy)):
                for j in range(len(entropy[0])):
                    unroll_entropy.append(entropy[i][j])
                    unroll_energy.append(energy[i][j])

            features = unroll_entropy + unroll_energy
            participant_trial.append(features)
            features_table.loc[len(features_table.index)] = features
            print(data_file)
            print(features)
        print(file)

    features_table.to_csv(directory + "features" + "db6" + ".csv", index=False)


def calc_wavelet_energy(data_set):
    # Входные параметры: 1 * N vector

    # Выходные параметры: Энергия вейвлета входного вектора, округляется до 3 знаков после запятой.

    wavelet_energy = np.nansum(np.log2(np.square(data_set)))

    return round(wavelet_energy, 3)


def calc_shannon_entropy(data_set):
    # Входные параметры: 1 * N vector

    # Выходные параметры: Энтропией вейвлета входного вектора, округляется до 3 знаков после запятой.

    probability = np.square(data_set)
    shannon_entropy = -np.nansum(probability * np.log2(probability))

    return round(shannon_entropy, 3)


def principal_component_analysis():
    data = pd.read_csv(directory + "features" + "db6" + ".csv")

    # 1. Нормализация данных и транспонирование
    normalised = pd.DataFrame(normalize(data, axis=0))

    # 2. Нахождение ковариационной матрицы
    covariance_df = normalised.cov()

    # 3. Собственные векторы
    u, s, v = np.linalg.svd(covariance_df)

    # 4. Основные компоненты
    data_reduced = normalised @ u
    data_reduced.head()

    data_reduced.to_csv(directory + "pc" + "db6" + ".csv", index=False)


def svm_classification():
    # Чтение данных и разделение
    pcs = pd.read_csv(directory + "pc" + "db6" + ".csv")

    outputs = pd.read_csv(directory + "outputs_main.csv", header=None)

    x = pcs.iloc[:, :].values
    y = outputs.iloc[:, :].values

    print(x.shape)
    print(x)
    print(y.shape)
    print(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Создание экземпляра модели
    svc = SVC()
    # Задание сетки параметров
    parameters_for_grid = {"C": (100, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9), "gamma": (1e-08, 1e-7, 1e-6, 1e-5)}
    # Создание объекта GridSearchCV
    grid_search = GridSearchCV(svc, parameters_for_grid, n_jobs=-1, cv=5)
    # Сопоставление объекта GridSearchCV с данными
    grid_search.fit(x_train, y_train)

    # Получение лучших параметров
    print(grid_search.best_params_)
    # Использования лучшей модели для прогноза
    svc_best = grid_search.best_estimator_
    # Получение лучшего результата
    accuracy = svc_best.score(x_test, y_test)
    print("Accuracy on the testing set is: {0:.1f}%".format(accuracy * 100))
    prediction = svc_best.predict(x_test)

    report = classification_report(y_test, prediction)
    print(report)


if __name__ == '__main__':
    preprocessing_data()
    # 1 step
    # feature_extraction()
    # 2 step
    #principal_component_analysis()
    # 3 step
    #svm_classification()
