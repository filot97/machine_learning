from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

import pandas as pd

if __name__ == "__main__":
    dataset = pd.read_csv("../data/data.csv")

    dummy_colums = ["X1", "X2", "X3", "X4", "X5", "X6", "X8",
                    "X9"]

    for column in dummy_colums:
        dataset = pd.concat([dataset, pd.get_dummies(dataset[column], prefix=column)], axis=1)
        dataset = dataset.drop(column, axis=1)

    x_columns = list(set(dataset.columns.tolist()) - set(["y"]))
    y_column = list(set(dataset.columns.tolist()) - set(x_columns))

    X = dataset.loc[:, x_columns].values
    y = dataset.loc[:, y_column].values

    X, y = RandomOverSampler().fit_sample(X, y)

    n_fold = 10
    skf = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=0)

    accuracy = []

    for train, test in skf.split(X, y):
        model = RandomForestClassifier(criterion="entropy", n_estimators=50, oob_score=False, random_state=0)
        model.fit(X[train], y[train])

        predicted = model.predict(X[test])
        k_accuracy = accuracy_score(y[test], predicted)
        accuracy.append(k_accuracy)

    print("\n %.f fold accuracy:" % n_fold, accuracy)