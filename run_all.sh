#!/usr/bin/env bash
# Reproduce every number, table and figure in the report from a clean state.
# Usage (inside the activated virtual environment):  bash run_all.sh
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"  # the oevl package, without pip install

echo "== 1/6 inputs"
if [[ ! -s data/brunetti-length.txt ]]; then python -m oevl.fetch_tauber; fi
if [[ ! -s data/ang_latn_broad.tsv ]]; then
  python - <<'PY'
from urllib.request import urlretrieve
from pathlib import Path
import hashlib, json, datetime
url = 'https://raw.githubusercontent.com/CUNY-CL/wikipron/master/data/scrape/tsv/ang_latn_broad.tsv'
p = Path('data/ang_latn_broad.tsv'); urlretrieve(url, p)
Path('data/wikipron_provenance.json').write_text(json.dumps({'url': url,
  'retrieved_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'sha256': hashlib.sha256(p.read_bytes()).hexdigest(), 'license': 'CC BY-SA 4.0 (Wiktionary data, see data/LICENSE.md; the WikiPron code is Apache-2.0)'}, indent=2))
PY
fi
sha256sum data/brunetti-length.txt data/ang_latn_broad.tsv 2>/dev/null || shasum -a 256 data/brunetti-length.txt data/ang_latn_broad.tsv
python -m oevl.audit_splits   # data/leakage_excluded.json: dev/test sentences sharing a 6-gram with train

echo "== 2/6 unit tests"
python -m unittest discover -s tests -v

echo "== 3/6 candidate forms: WikiPron, then Clark Hall + Wiktionary tables + Bosworth-Toller"
python -m oevl.wikipron_to_marked data/ang_latn_broad.tsv data/lex_wikipron.csv
if [[ -s data/lex_clarkhall.csv && -s data/lex_wiktionary.csv && -s data/lex_bt.csv ]]; then
  python -m oevl.build_lexicons --unions-only      # the three converted lists ship in data/
else
  python -m oevl.build_lexicons                    # downloads the three sources into data/raw/ and converts them
fi

echo "== 4/6 experiments (one set per run: the John dev set and the two test sets)"
# PLAN D3, decided on dev: the four settings train without Zupitza, and results/wikipron_with_zupitza keeps all five works
D3="--exclude-train zupitza_grammar"
rm -rf results/training_only results/wikipron results/proposal results/all results/wikipron_with_zupitza
for t in john_ws soliloquies tauber_beowulf; do
  python -m oevl.experiment --data data --tauber data/brunetti-length.txt --out results/training_only --tests "$t" $D3 > "results_training_only_$t.log"
  for lex in wikipron proposal all; do
    python -m oevl.experiment --data data --tauber data/brunetti-length.txt --out "results/$lex" --lexicon "data/lex_$lex.csv" --tests "$t" $D3 > "results_${lex}_$t.log"
  done
  python -m oevl.experiment --data data --tauber data/brunetti-length.txt --out results/wikipron_with_zupitza --lexicon data/lex_wikipron.csv --tests "$t" > "results_wikipron_with_zupitza_$t.log"
  echo "   done: $t"
done
for d in training_only wikipron proposal all wikipron_with_zupitza; do mv results_${d}_{john_ws,soliloquies,tauber_beowulf}.log "results/$d/"; done

if [[ -s results/charsiu/ipa.tsv ]]; then
  python -m oevl.charsiu_g2p analyse > results/charsiu/analyse.log   # uses the shipped CharsiuG2P outputs (made on Colab)
fi
mkdir -p results/agreement
python -m oevl.agreement > results/agreement/agreement.log   # three Beowulf editions, kappa and the S7 gate

echo "== 5/6 check all headline values"
python -m oevl.check_expected

echo "== 6/6 report tables, numbers and figure"
python -m oevl.error_typing   # error typing table from the adjudicated labels in results/error_typing/labels.tsv
python -m oevl.encoder_summary --from-tsv   # encoder table and numbers from the committed GPU predictions (no GPU)
python -m oevl.select_config   # the encoder selection rule on the committed dev runs -> results/encoder/selection.json
python -m oevl.build_report
