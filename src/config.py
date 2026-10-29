from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 10

# TF-IDF / BoW Settings
MAX_FEATURES = 9000
MIN_DF = 3

# Classifier Hyperparameters
NB_ALPHA = 1.0
SVM_C = 1.0
MLP_EPOCHS = 15
MLP_BATCH_SIZE = 32
MLP_LR = 0.001

DISTILBERT_EPOCHS = 3
DISTILBERT_BATCH_SIZE = 16
DISTILBERT_LR = 2e-5

DATASETS = ["spamassassin", "enron", "sms"]
MODELS = ["nb", "svm", "mlp", "distilbert"]
FEATURES = ["bow", "tfidf"]
