# CSCI323 Group Project: Email Spam Detection

We compare two classic spam filters, **Naive Bayes** and a linear **Support Vector Machine**, using two text representations (**Bag of Words** and **TF-IDF**). Then we test how well they hold up when:

1. they are tested on a different dataset from the one they were trained on (**dataset shift**), and
2. spammers disguise their messages (**adversarial attacks**).

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

## Experiments

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

## Datasets

| Name | What it is | Size |
|---|---|---|
| SpamAssassin | Apache public corpus, emails from 2002–2005 | ~6k emails |
| Enron-Spam | Enron employee mail plus spam (Metsis et al., 2006) | ~33k emails |
| SMS Spam | UCI SMS Spam Collection: text messages, not emails | ~5.5k messages |

`python main.py download` fetches all three into `data/raw/`. That folder is not committed to Git.

## Code layout

```
main.py               command-line entry point
src/config.py         all settings (seed, vocabulary size, SVM C, ...)
src/download_data.py  dataset download
src/load_data.py      parse each dataset into text + label
src/preprocess.py     text cleaning
src/models.py         BoW / TF-IDF + Naive Bayes / SVM pipelines
src/evaluate.py       metrics and plots
src/adversarial.py    obfuscation and good-word attacks
src/experiments.py    the five experiments
tests/smoke_test.py   quick test on fake data
```
