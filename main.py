# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import librosa

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA

from ImportSpeech import *
from WaveVisualize import *
from DataAugmentation import *
from FeatureExtraction import *


def concat_data(*d):
    d_path = pd.concat(d, axis=0)
    d_path.to_csv("data_path.csv", index=False)


def data_visualisation():
    plt.suptitle('Count of Emotions')
    sns.countplot(x=data_path.Emotions)
    plt.ylabel('Count', size=12)
    plt.xlabel('Emotions', size=12)
    sns.despine(top=True, right=True, left=False, bottom=False)
    plt.show()


def analyse_signal(emotion):
    path = np.array(data_path.Path[data_path.Emotions == emotion])[1]
    data, sampling_rate = librosa.load(path)

    # applying any technique to change data
    # - noise
    # - stretch
    # - shift
    # - pitch
    # data = pitch(data, sampling_rate)

    create_waveplot(data, sampling_rate, emotion)
    create_spectrogram(data, sampling_rate, emotion)


def feature_extract():
    X, Y = [], []
    banned = ["surprise"]
    for path, emotion in tqdm.tqdm(zip(data_path.Path, data_path.Emotions)):
        if emotion not in banned:
            try:
                # print(emotion)
                feature = get_features(path)
                for i, ele in enumerate(feature):
                    X.append(ele)
                    Y.append(emotion)
            except:
                print(emotion, path, "Error")

    print(len(X), len(Y), data_path.Path.shape)
    Features = pd.DataFrame(X)
    Features['labels'] = Y
    Features.to_csv('features.csv', index=False)


def SVM():
    from sklearn import svm

    # clf = svm.SVC(C=1200)
    clf = svm.SVC(C=500, decision_function_shape='ovo', gamma=0.0001)
    clf.fit(X_train, Y_train)
    print("Train:", clf.score(X_train, Y_train))  # 0.7757384882710686 (250),
    print("Test:", clf.score(X_test, Y_test))  # 0.6249637995945555 (250),

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(estimator=clf, X=X, y=label_encoder_y, cv=kfold, n_jobs=-1,
                            scoring=['accuracy'])

    print('Training 5-fold Cross Validation Results:\n')
    print('Accuracy: ', scores['test_accuracy'].mean())  # 0.6209962351578338

    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, Y_test, display_labels=Features.labels.unique(),
                                                     cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    # plt.figure((12, 12))
    plt.show()


def KNN():
    from sklearn.neighbors import KNeighborsClassifier

    neigh = KNeighborsClassifier(n_neighbors=20, weights='distance', leaf_size=5, p=2)
    neigh.fit(X_train, Y_train)

    unique, counts = np.unique(Y_train, return_counts=True)

    result = np.column_stack((unique, counts))
    print(result)

    print(neigh.score(X_test, Y_test))  # 0.5645815233130611
    print(neigh.score(X_train, Y_train))  # 0.9998551983782218

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(estimator=neigh, X=X, y=label_encoder_y, cv=kfold, n_jobs=-1,
                            scoring=['accuracy'])

    print('Training 5-fold Cross Validation Results:\n')
    print('Accuracy: ', scores['test_accuracy'].mean())  # 0.5622357370402549

    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(neigh, X_test, Y_test, display_labels=Features.labels.unique(),
                                                     cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    # plt.figure((12, 12))
    plt.show()


def DT():
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(max_depth=25, min_samples_leaf=2, criterion='entropy', random_state=0)
    clf.fit(X_train, Y_train)

    print(clf.get_n_leaves())  # 4261
    print(clf.score(X_test, Y_test))  # 0.5774688676513177
    print(clf.score(X_train, Y_train))  # 0.9557269041413263

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    scores = cross_validate(estimator=clf,
                            X=X,
                            y=label_encoder_y,
                            cv=kfold,
                            n_jobs=-1,
                            scoring=['accuracy'])

    print('Training 5-fold Cross Validation Results:\n')
    print('Accuracy: ', scores['test_accuracy'].mean())  # 0.5808861859252824

    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(clf, X_test, Y_test, display_labels=Features.labels.unique(),
                                                     cmap=plt.cm.Blues, normalize=normalize)
        disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)

    # plt.figure((12, 12))
    plt.show()


if __name__ == '__main__':
    file_data_path = "data_path.csv"

    if not os.path.isfile(file_data_path):
        Ravdess_df = import_ravdess()
        Crema_df = import_crema()
        Tess_df = import_tess()
        Savee_df = import_savee()
        print("End import")
        concat_data(Ravdess_df, Crema_df, Tess_df, Savee_df)
    print("end")

    data_path = pd.read_csv("data_path.csv")
    data_visualisation()

    # fear  angry  surprise  calm  sad  happy disgust
    # analyse_signal('fear')

    # feature_extract()
    # Features = pd.read_csv('features.csv')
    # X = Features.iloc[:, :-1].values
    # Y = Features['labels'].values
#
    # label_encoder = LabelEncoder()
    # label_encoder = label_encoder.fit(Y)
    # label_encoder_y = label_encoder.transform(Y)

    # pca = PCA(n_components=144)
    # X = pca.fit_transform(X)

    # X_train, X_test, Y_train, Y_test = train_test_split(X, label_encoder_y, test_size=0.2, random_state=0)
    # SVM()
    # KNN()
    # DT()
