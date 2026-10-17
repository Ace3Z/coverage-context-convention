#!/usr/bin/env bash
# Reproduce every number, table and figure in the report from a clean state.
# Usage (inside the activated virtual environment):  bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

echo "== 1/6 inputs"
if [[ ! -s data/brunetti-length.txt ]]; then python src/fetch_tauber.py; fi
if [[ ! -s data/ang_latn_broad.tsv ]]; then
  python - <<'PY'
from urllib.request import urlretrieve
from pathlib import Path
import hashlib, json, datetime
url = 'https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv/ang_latn_broad.tsv'
p = Path('data/ang_latn_broad.tsv'); urlretrieve(url, p)
Path('data/wikipron_provenance.json').write_text(json.dumps({'url': url,
  'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'license': 'Apache-2.0'}, indent=2))
PY
fi
sha256sum data/brunetti-length.txt data/ang_latn_broad.tsv 2>/dev/null || shasum -a 256 data/brunetti-length.txt data/ang_latn_broad.tsv

echo "== 2/6 unit tests"
python -m unittest discover -s tests -v

echo "== 3/6 candidate forms: WikiPron, then Clark Hall + Wiktionary tables + Bosworth-Toller"
python src/wikipron_to_marked.py data/ang_latn_broad.tsv data/lex_wikipron.csv
if [[ -s data/lex_clarkhall.csv && -s data/lex_wiktionary.csv && -s data/lex_bt.csv ]]; then
  python src/build_lexicons.py --unions-only      # the three converted lists ship in data/
else
  python src/build_lexicons.py                    # downloads the three sources into data/raw/ and converts them
fi

echo "== 4/6 experiments (one test set per run)"
rm -rf results_training_only results_wikipron results_proposal results_all
for t in soliloquies tauber_beowulf; do
  python src/experiment.py --data data --tauber data/brunetti-length.txt --out results_training_only --tests "$t" > "results_training_only_$t.log"
  for lex in wikipron proposal all; do
    python src/experiment.py --data data --tauber data/brunetti-length.txt --out "results_$lex" --lexicon "data/lex_$lex.csv" --tests "$t" > "results_${lex}_$t.log"
  done
  echo "   done: $t"
done
for d in training_only wikipron proposal all; do mv results_${d}_*.log "results_$d/"; done

if [[ -s results_charsiu/ipa.tsv ]]; then
  python src/charsiu_g2p.py analyse > results_charsiu/analyse.log   # uses the shipped CharsiuG2P outputs (made on Colab)
fi

echo "== 5/6 check all headline values"
python src/check_expected.py

echo "== 6/6 report tables, numbers and figure"
python src/build_report.py
