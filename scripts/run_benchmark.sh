#!/usr/bin/env bash
# Run every experiment: both models, in-domain + the 3 leave-one-dataset-out tests.
# Extra arguments go to the DistilBERT runs, e.g.  ./run_all.sh --max-train 6000
set -e
cd "$(dirname "$0")"
python download_data.py
for h in "" sms enron spamassassin; do
  python train_mlp.py ${h:+--holdout $h}
done
for h in "" sms enron spamassassin; do
  python train_distilbert.py ${h:+--holdout $h} "$@"
done
python summarize_results.py
