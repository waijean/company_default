import os

import git
from sklearn.model_selection import StratifiedKFold

repo = git.Repo("../utils", search_parent_directories=True)
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
RUN_ID = "run_id"

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


RANDOM_STATE = 42
TRAIN_RATIO_SET = "ratio_train.csv"  # "cleaned_ratio_train.csv"
TRAIN_RAW_SET = "raw_train.csv"  # "cleaned_raw_train.csv"
TEST_RATIO_SET = "ratio_test.csv"  # "cleaned_ratio_test.csv"
TEST_RAW_SET = "raw_test.csv"  # "cleaned_raw_test.csv"
TRAIN_COM_SET = "combined_train.csv"
TEST_COM_SET = "combined_test.csv"

FEATURE_LIST = [
    "sales (n) / sales (n-1)",
    "PREVIOUS_YEAR_SALES",
    "EXTRAORDINARY_ITEMS",
    "FINANCIAL_EXPENSES",
    "(gross profit + depreciation) / total liabilities",
    "profit on operating activities / financial expenses",
    "(net profit + depreciation) / total liabilities",
    "(current assets - inventory) / short-term liabilities",
    "gross profit (in 3 years) / total assets",
    "GROSS_PROFIT_IN_3_YEARS",
    "(gross profit + depreciation) / sales",
    "net profit / total assets",
    "profit on sales / total assets",
    "SHARE_CAPITAL",
    "[(cash + short-term securities + receivables - short-term liabilities) / (operating expenses - depreciation)] * 365",
    "PROFIT_ON_SALES",
    "total liabilities / ((profit on operating activities + depreciation) * (12/365))",
    "DEPRECIATION",
    "profit on sales / sales",
    "(equity - share capital) / total assets",
    # "gross profit / short-term liabilities",
    # "PROFIT_ON_OPERATING_ACTIVITIES",
    # "SHORT_TERM_SECURITIES",
    # "NET_PROFIT",
    # "INTEREST",
    # "(total liabilities * 365) / (gross profit + depreciation)",
    # "net profit / inventory",
    # "operating expenses / total liabilities",
    # "RETAINED_EARNINGS",
    # "(sales - cost of products sold) / sales",
    # "sales / total assets",
    # "profit on operating activities / sales",
    # "OPERATING_EXPENSES",
    # "(receivables * 365) / sales",
    # "total costs /total sales",
    # "EBIT / total assets",
    # "EBIT",
    # "(gross profit + interest) / sales",
    # "retained earnings / total assets",
    # "sales / receivables",
    # "EBITDA",
    # "WORKING_CAPITAL",
    # "profit on operating activities / total assets",
    # "constant capital / total assets",
    # "(current assets - inventory - short-term liabilities) / (sales - gross profit - depreciation)",
    # "BOOK_VALUE_OF_EQUITY",
    # "EBITDA (profit on operating activities - depreciation) / total assets",
    # "working capital / fixed assets",
    # "TOTAL_COSTS",
    # "gross profit / total assets",
]
