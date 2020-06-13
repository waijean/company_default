import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from imblearn.datasets import make_imbalance
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced

from model_utils import model_pipeline_io

RANDOM_STATE = 0


def run():
    train_data = model_pipeline_io.read_train_file()
    X, y = make_imbalance(
        train_data.iloc[:, :-1],
        train_data.iloc[:, -1],
        sampling_strategy={0: 500, 1: 500},
        random_state=RANDOM_STATE,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    pipe = make_pipeline(
        NearMiss(version=2),
        StandardScaler(),
        RandomForestClassifier(random_state=RANDOM_STATE),
    )
    pipe.fit(X_train, y_train)
    score = pipe.score(X_test, y_test)

    print(f"model accuracy on test set: {score}")
    print(classification_report_imbalanced(y_test, pipe.predict(X_test)))


if __name__ == "__main__":
    print(os.path.abspath(""))
    run()
