# GPU notebooks (experimental pilots)

Four notebooks are supplied: Kaggle and Colab, each with a 128-byte and 256-byte input-cap variant. Each runs five arms (sentence, word, left, local ±2, local ±5), 10,000 seeded training examples, one epoch, batch 32, microbatch 8, LR 3e-4, seed 1. **Only the input cap differs between paired 128/256 pilots.** These are exploratory, not full-data results.

The notebooks expect the **updated project** (including `src/encoder_context_windows.py` and `--train-limit` in `src/encoder_experiment.py`). Upload this release to Kaggle as a dataset or Colab as a zip. GPU availability and session limits are external constraints. Run 128 first; run 256 only if needed for a controlled input-length comparison. Do not run both simultaneously against the same output directory.

**IMPORTANT:** The original Kaggle and Colab notebooks were adapted from previously supplied notebooks. Syntax was checked, but GPU execution, full setup, and end-to-end training have **not** been tested. Validate setup and smoke test before a long run. No mid-experiment checkpoints. The runner creates a ZIP after each completed arm; download it promptly.
