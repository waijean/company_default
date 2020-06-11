import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from model_utils import model_pipeline_io


def run():
    train_data = model_pipeline_io.read_train_file()
    X_train, X_test, y_train, y_test = train_test_split(
        train_data.iloc[:, :-1], train_data.iloc[:, -1], random_state=0
    )
    pipe = Pipeline([("scaler", StandardScaler()), ("rf", RandomForestClassifier())])
    pipe.fit(X_train, y_train)
    pipe.score(X_test, y_test)


if __name__ == "__main__":
    print(os.path.abspath(""))
    run()
