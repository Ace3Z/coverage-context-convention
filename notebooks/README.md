# GPU notebooks

The ByT5 pilot needs a GPU, so it runs on Kaggle or Colab. Both notebooks only put the project on the machine,
install `requirements.txt` and call `python -m oevl.pilot`. The logic, the checks and the resume behaviour live in
`src/oevl/pilot.py`. To make the project zip, run `git archive -o oe_project.zip HEAD` in the repository.

| Notebook | Platform | How the project gets there |
|---|---|---|
| `pilot_kaggle.ipynb` | Kaggle, T4 GPU | the repository folder or a zip of it, added as an input dataset |
| `pilot_colab.ipynb` | Colab, T4 GPU | a zip of the repository, uploaded when the first cell asks |

Set `MAX_BYTES` (128 or 256) in the first code cell. Everything else is fixed: five arms (sentence, word, left, local2,
local5), the same 10,000 training examples for every arm, one epoch, batch 32, microbatch 8, learning rate 3e-4,
seed 1, and all five training works (the pilots ran before decision D3 left Zupitza out).
Run 128 first. Compare a 256 run only with the 128 run of the same arms.

After each finished arm the results folder is packed into `encoder_pilot_<bytes>_results.zip`, which you should
download. To resume in a new session on Kaggle, add that zip, a renamed copy such as `... (1).zip`, or its extracted
folder as input. On Colab, upload the zip to `/content`. Finished arms are copied in, checked and skipped, and an arm
that was interrupted starts again from the beginning. No new arm starts after 9 hours of the notebook session,
counted from the first cell, so an arm cannot run into Kaggle's 12-hour limit.

The pilots are exploratory. With only 10,000 examples and one epoch, an arm's score also reflects how fast it learns,
so the pilot's ranking of arms need not match the full-data ranking. A full CPU dry run of the same loop:

    PYTHONPATH=src python -m oevl.pilot --tiny --allow-cpu --skip-model-check --max-bytes 128 --out /tmp/pilot
