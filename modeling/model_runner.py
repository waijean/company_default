import pandas as pd
import numpy as np
from sklearn.preprocessing import QuantileTransformer

from imblearn.pipeline import make_pipeline, Pipeline
from imblearn.ensemble import BalancedRandomForestClassifier

from conf.tables import TRAIN_COMBINED_SET, TEST_COMBINED_SET
from modeling.utils import model_pipeline_io
from config import hyperparameter


def train_pipeline(train_set_name: list) -> Pipeline:
    """
    Train the pipeline on the full training set given
    """
    X_train, y_train = model_pipeline_io.get_training_set(train_set_name)
    # check that there is no infinite values in X_train
    assert all(np.isinf(X_train).sum() == 0)
    pipeline = make_pipeline(
        QuantileTransformer(),
        BalancedRandomForestClassifier(**hyperparameter.COMBINE_DATA),
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def predict(test_set_name: list, pipeline: Pipeline) -> pd.DataFrame:
    """
    Use the trained pipeline to predict on the test set given and output the final prediction as submission file
    """
    X_test = model_pipeline_io.get_test_set(test_set_name)

    y_pred = pipeline.predict(X_test)
    submission = pd.DataFrame(data=y_pred)

    model_pipeline_io.save_submit_file(submission, "submission.csv")
    return submission


if __name__ == "__main__":
    train_set_name = [TRAIN_COMBINED_SET]
    pipeline = train_pipeline(train_set_name)

    test_set_name = [TEST_COMBINED_SET]
    submission = predict(test_set_name, pipeline)
