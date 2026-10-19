# GPU notebooks

The encoder pilot needs a GPU, so it runs on Kaggle or Colab. Both notebooks only put the project on the machine,
install `requirements.txt` and call `python -m oevl.pilot`; the logic, checks and resume behaviour are in
`src/oevl/pilot.py`. Make the project zip in the repository with `git archive -o oe_project.zip HEAD`.

| Notebook | Platform | How the project gets there |
|---|---|---|
| `pilot_kaggle.ipynb` | Kaggle, GPU T4 | the repository folder or a zip of it, added as an input dataset |
| `pilot_colab.ipynb` | Colab, T4 GPU | a zip of the repository, uploaded when the first cell asks |

Set `MAX_BYTES` (128 or 256) in the first code cell. Everything else is fixed: five arms (sentence, word, left, local2,
local5), the same 10,000 training examples for every arm, one epoch, batch 32, microbatch 8, learning rate 3e-4,
seed 1. Run 128 first. The 256 run answers a question only if it is compared with the 128 run of the same arms.

After every finished arm the results folder is packed into `encoder_pilot_<bytes>_results.zip`. Download it. To
resume in a new session, add that zip, a renamed copy such as `... (1).zip`, or its extracted folder as input (Kaggle),
or upload the zip to `/content` (Colab). Finished arms are copied in, validated and skipped; an arm that was interrupted
starts again from the beginning. No new arm starts after 9 hours of the notebook session, counted from the first cell,
so an arm cannot run into Kaggle's 12-hour limit.

These pilots are exploratory. With only 10,000 examples and one epoch, an arm's score also reflects how fast it learns,
so do not read the pilot's ranking of arms as the full-data ranking. A full CPU dry run of the same loop:

    PYTHONPATH=src python -m oevl.pilot --tiny --allow-cpu --skip-model-check --max-bytes 128 --out /tmp/pilot
