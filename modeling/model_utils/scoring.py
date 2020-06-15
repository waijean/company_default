from sklearn.metrics import (
    confusion_matrix,
    make_scorer,
    precision_score,
    recall_score,
    f1_score,
)

from model_utils.constants import (
    ACCURACY,
    PRECISION,
    RECALL,
    F1,
    TP,
    TN,
    FP,
    FN,
    BALANCED_ACCURACY,
)


def tp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 0]


def tn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 1]


def fp(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[1, 0]


def fn(y_true, y_pred):
    return confusion_matrix(y_true, y_pred)[0, 1]


BINARY_CLASSIFIER_SCORING = {
    ACCURACY: ACCURACY,
    BALANCED_ACCURACY: BALANCED_ACCURACY,
    PRECISION: make_scorer(precision_score, average="binary"),
    RECALL: make_scorer(recall_score, average="binary"),
    F1: make_scorer(f1_score, average="binary"),
    TP: make_scorer(tp),
    TN: make_scorer(tn),
    FP: make_scorer(fp),
    FN: make_scorer(fn),
}


MICRO_CLASSIFIER_SCORING = {
    ACCURACY: ACCURACY,
    PRECISION: make_scorer(precision_score, average="micro"),
    RECALL: make_scorer(recall_score, average="micro"),
    F1: make_scorer(f1_score, average="micro"),
}
