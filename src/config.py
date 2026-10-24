from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10

# same as the paper
MAX_FEATURES = 9000
MIN_DF = 3

NB_ALPHA = 1.0
SVM_C = 1.0  # paper uses 1000

DATASETS = ["spamassassin", "enron", "sms"]
MODELS = ["nb", "svm"]
FEATURES = ["bow", "tfidf"]
