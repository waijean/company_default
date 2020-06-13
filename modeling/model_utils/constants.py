import os

import git
from sklearn.model_selection import StratifiedKFold

repo = git.Repo(".", search_parent_directories=True)
ROOT_DIR_PATH = repo.working_tree_dir

LOG_CONFIG_PATH = os.path.join(ROOT_DIR_PATH, "conf/logging.conf")

# mlrun path
TRACKING_URI_PATH = "file:" + os.path.join(ROOT_DIR_PATH, "mlruns")
ARTIFACT_PATH = "file:" + os.path.join(ROOT_DIR_PATH, "mlruns")
TEST_EXPERIMENT_NAME = "Pytest"
TEST_RUN_NAME = "DecisionTree"

# mlrun tags
X_COL = "X_col"
Y_COL = "y_col"
iris_X_COL = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]
iris_y_COL = "target"

# mlrun artifacts
PIPELINE_HTML = "pipeline.html"
SCORES_CSV = "scores.csv"
FEATURE_IMPORTANCE_CSV = "feature_importance.csv"
FEATURE_IMPORTANCE_PLOT = "feature_importance_plot"

# mlrun metrics
ACCURACY = "accuracy"
BALANCED_ACCURACY = "balanced_accuracy"
PRECISION = "precision"
RECALL = "recall"
F1 = "f1"
TP = "tp"
TN = "tn"
FP = "fp"
FN = "fn"


# cross validate default
DEFAULT_CV = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
