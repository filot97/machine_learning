from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras import models
from keras import layers
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import tensorflow as tf
import numpy as np

class DeepLearningModel():
    def __init__(self, file_name=None):
        self.file_name = file_name

    def label_eccoding(self, column):
        encoder = LabelEncoder()

        column = encoder.fit_transform(column)
        column = np_utils.to_categorical(column)

        return column

    def load_data(self):
        dataset = pd.read_csv(self.file_name)

        dummy_colums = ["X1", "X2", "X3", "X4", "X5", "X6", "X8", "X9"]

        for column in dummy_colums:
            dataset = pd.concat([dataset, pd.get_dummies(dataset[column], prefix=column)], axis=1)
            dataset = dataset.drop(column, axis=1)

        x_columns = list(set(dataset.columns.tolist()) - set(["y"]))
        y_column = list(set(dataset.columns.tolist()) - set(x_columns))

        X = dataset.loc[:, x_columns].values
        y = dataset.loc[:, y_column].values

        X, y = RandomOverSampler().fit_sample(X, y)
        y = self.label_eccoding(y)

        return X, y

    def create_model(self):
        X, y = self.load_data()

        model = models.Sequential()
        model.add(layers.Dense(units=1000, activation="relu", input_dim=X.shape[1]))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(units=y.shape[1], activation="relu"))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(units=y.shape[1], activation="softmax"))

        model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

        return model

if __name__ == "__main__":
    line_predictor = DeepLearningModel("../data/data.csv")

    seed = 0
    np.random.seed(seed)
    tf.set_random_seed(seed)

    n_fold = 10
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=seed)

    accuracy = []
    X, y = line_predictor.load_data()

    for train, test in skf.split(X, y.argmax(1)):
        model = line_predictor.create_model()
        model.fit(X[train], y[train], epochs=1000, batch_size=200, verbose=2)

        k_accuracy = "%.4f" % (model.evaluate(X[test], y[test])[1])
        accuracy.append(k_accuracy)

    # 결과 출력
    print("\n %.f fold accuracy:" % n_fold, accuracy)