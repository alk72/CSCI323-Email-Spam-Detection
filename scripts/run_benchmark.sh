#!/usr/bin/env bash
# Shared-split benchmark: NB, SVM, MLP and DistilBERT, in-domain + the 3 leave-one-dataset-out tests.
# Extra arguments go to the DistilBERT runs, e.g.  scripts/run_benchmark.sh --max-train 6000
set -e
cd "$(dirname "$0")/.."
python -m src.benchmark.prepare_data
for h in "" sms enron spamassassin; do
  python -m src.benchmark.train_classical ${h:+--holdout $h}
  python -m src.benchmark.train_mlp ${h:+--holdout $h}
done
python -m src.benchmark.train_classical --balance
for h in "" sms enron spamassassin; do
  python -m src.benchmark.train_distilbert ${h:+--holdout $h} "$@"
done
python -m src.benchmark.summarize
