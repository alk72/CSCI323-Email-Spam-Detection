# CSCI323 Group Project: Email Spam Detection

We compare four spam filters on three public datasets:

- two classic models, **Naive Bayes** and a linear **Support Vector Machine**, each with **Bag of Words** or **TF-IDF** features, and
- two neural models, a **Multi-Layer Perceptron** on TF-IDF features and a fine-tuned **DistilBERT** transformer.

Then we test how well they hold up when:

1. they are tested on a different dataset from the one they were trained on (**dataset shift**), and
2. spammers disguise their messages (**adversarial attacks**).

The code has two parts that share settings, data folders and the NB/SVM pipeline:

| Part | Entry point | Models | What it is for |
|---|---|---|---|
| Classical experiments | `main.py` | NB, SVM | Cross-validation, cross-dataset, margin and attack analysis |
| Shared-split benchmark | `src/benchmark/` | NB, SVM, MLP, DistilBERT | All four models trained and tested on one fixed 70/15/15 split, so they can go in one table |

The project builds on this paper:

> Jáñez-Martino, F., Alaiz-Rodríguez, R., González-Castro, V., Fidalgo, E., & Alegre, E. (2023). A review of spam email detection: analysis of spammer strategies and the dataset shift problem. *Artificial Intelligence Review, 56*, 1145–1173. https://doi.org/10.1007/s10462-022-10195-4

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python main.py download         # ~25 MB, first time only
python main.py all              # runs every experiment, a few minutes
```

Tables are saved to `results/tables/` and figures to `results/figures/`.

## Classical experiments (`main.py`)

| Command | What it does | Main output |
|---|---|---|
| `python main.py explore` | Dataset sizes, spam share, message length, common words | `dataset_stats.csv`, `dataset_overview.png` |
| `python main.py cv` | 10-fold cross-validation within each dataset (4 models x 3 datasets) | `cv_results.csv`, `cv_results.png` |
| `python main.py cross` | Train on one dataset and test on each of the others | `cross_dataset.csv`, `cross_accuracy.png`, `cross_fpr.png` |
| `python main.py margin` | Where errors happen relative to the decision boundary; borderline emails; most important words | `margin_*.csv`, `margin_*.png`, `borderline_emails_*.csv` |
| `python main.py attack` | Spam recall under word obfuscation and good-word injection | `attack_*.csv`, `attack_*.png` |

Useful options:
- `--datasets spamassassin enron` runs on fewer datasets.
- `--target enron` picks the dataset for `margin` and `attack`.

Run `python -m tests.smoke_test` to check that the code works without downloading anything.

## Shared-split benchmark (`src/benchmark/`)

```bash
pip install -r requirements-neural.txt          # adds PyTorch + Transformers for DistilBERT
python -m src.benchmark.prepare_data            # downloads if needed -> data/processed/spam_all.csv

python -m src.benchmark.train_classical         # NB + SVM, BoW + TF-IDF
python -m src.benchmark.train_mlp               # TF-IDF + MLP, a few minutes on a laptop CPU
python -m src.benchmark.train_distilbert        # GPU recommended (free Colab T4 works)
python -m src.benchmark.summarize               # one comparison table -> results/benchmark/results_summary.csv
python -m src.benchmark.marginal_cases          # margin, attack and edge-case tests for all four models

python -m src.benchmark.predict --model mlp "WINNER!! claim your prize now"   # demo
```

Add `--holdout sms` (or `enron`, `spamassassin`) to any `train_*` command for the cross-dataset test: the model never sees that dataset during training and is then tested on all of it. `scripts/run_benchmark.sh` runs every combination, and `notebooks/colab_run.ipynb` does the same on Google Colab. Run `python -m tests.smoke_test_benchmark` for a quick check on fake data.

Each run saves to `results/benchmark/<model>_<experiment>/`: `metrics.json` (accuracy, precision, recall, F1, ROC-AUC, confusion counts, per-dataset breakdown), `history.csv` (loss and validation F1 per epoch) and the trained model.

### How the benchmark data is prepared

- E-mail headers are removed (only Subject and body are kept) and HTML is converted to text, so a model can't learn shortcuts from server names or dates.
- URLs become `URL`, e-mail addresses `EMAIL`, long numbers `PHONENUMBER`. A spam campaign's exact link never comes back, but "contains a link" does.
- Exact duplicates are removed, and messages that appear with both labels are dropped.
- Each dataset is split 70% train / 15% validation / 15% test, stratified by label, with seed 42.
- Training samples every dataset equally per epoch (turn off with `--no-balance`), so Enron, the biggest, doesn't drown out the short SMS style. Model selection uses the mean validation F1 across datasets.

### The two neural models

| | MLP | DistilBERT |
|---|---|---|
| Input | TF-IDF (word 1-2 grams + character 3-5 grams, 50k features) | WordPiece tokens (up to 256) |
| Network | 1 hidden layer (128 ReLU units) + softmax | 6-layer pretrained transformer + classification head on `[CLS]` |
| Parameters | about 6.4M | about 66M |
| Training | From scratch: Adam, cross-entropy, L2 penalty, early stopping | Fine-tuning: AdamW, lr 2e-5, warm-up + linear decay, weight decay, dropout, early stopping |

## Datasets

| Name | What it is | Size |
|---|---|---|
| SpamAssassin | Apache public corpus, emails from 2002–2005 | ~6k emails |
| Enron-Spam | Enron employee mail plus spam (Metsis et al., 2006) | ~33k emails |
| SMS Spam | UCI SMS Spam Collection: text messages, not emails (Almeida et al., 2011) | ~5.5k messages |

`python main.py download` fetches all three into `data/raw/`, and `python -m src.benchmark.prepare_data` reuses the same files. Neither `data/` nor `results/` is committed to Git.

## Code layout

```
main.py                         command-line entry point for the classical experiments
src/config.py                   all settings (seed, paths, vocabulary size, SVM C, ...)
src/download_data.py            dataset download
src/load_data.py                parse each dataset into text + label
src/preprocess.py               text cleaning
src/models.py                   BoW / TF-IDF + Naive Bayes / SVM pipelines
src/evaluate.py                 metrics and plots
src/adversarial.py              obfuscation and good-word attacks
src/experiments.py              the five classical experiments
src/benchmark/common.py         benchmark cleaning, TF-IDF features, splits, metrics
src/benchmark/prepare_data.py   build the shared 70/15/15 dataset
src/benchmark/train_classical.py  NB / SVM on the shared split
src/benchmark/train_mlp.py      TF-IDF + MLP
src/benchmark/train_distilbert.py  DistilBERT fine-tuning
src/benchmark/summarize.py      collect all runs into one table
src/benchmark/marginal_cases.py  margin, attack and edge-case tests for all four models
src/benchmark/np_distilbert.py  NumPy DistilBERT forward pass, used when PyTorch is missing
src/benchmark/predict.py        classify your own messages (demo)
scripts/run_benchmark.sh        run the whole benchmark
notebooks/colab_run.ipynb       run the benchmark on Google Colab
tests/smoke_test.py             classical experiments on fake data
tests/smoke_test_benchmark.py   benchmark on fake data
```
