"""Exploratory GPU pilot: the ByT5 encoder's context arms on a fixed training subset.

This is the logic that used to live in the four Kaggle/Colab notebooks. The notebooks in notebooks/ now only put
the project on the machine and call the line below, adding --max-hours with the hours left of their 9-hour budget:

  python -m oevl.pilot --max-bytes 128 --out /kaggle/working/encoder_pilot_128_results [--restore OLD.zip ...]

Steps: check the GPU and the pinned transformers version, load the checkpoint, run a tiny CPU-sized smoke test,
copy in the finished arms of any --restore, then train and evaluate each arm in its own process. A finished arm is
validated, marked COMPLETE.json and packed into <out>.zip, so a later session resumes with --restore <zip or folder>
and skips it. There is no checkpoint inside an arm: an arm that is interrupted starts again from the beginning.

Defaults are the agreed pilot: arms sentence, word, left, local2, local5 (P1-P5), 10,000 training examples,
1 epoch, batch 32, microbatch 8, learning rate 3e-4, seed 1. Every arm trains on all five training works, because
the pilots ran before decision D3 left Zupitza's Grammar out. Only --max-bytes differs between the 128 and 256
pilots. CPU dry run of the whole loop: --tiny --allow-cpu --skip-model-check.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
# pilot arm name to experiment id (P1 to P5), which also names the arm's folder under --out
ARMS = {"sentence": "P1", "word": "P2", "left": "P3", "local2": "P4", "local5": "P5"}
# the test sets every arm is checked and summarised on (encoder_experiment also scores the dev set john_ws)
TESTS = ("soliloquies", "tauber_beowulf")


def env() -> dict[str, str]:
    """The environment for the child processes: a copy of os.environ with one OpenBLAS, OpenMP and MKL thread.

    It also turns tokenizer parallelism off and puts <repo>/src first on PYTHONPATH, keeping an existing PYTHONPATH
    after it."""
    e = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
             TOKENIZERS_PARALLELISM="false")
    e["PYTHONPATH"] = os.pathsep.join(x for x in (str(ROOT / "src"), e.get("PYTHONPATH")) if x)
    return e


def required_transformers() -> str | None:
    """The transformers version pinned by a transformers==X line in requirements.txt, or None if there is none."""
    m = re.search(r"^transformers==(\S+)", (ROOT / "requirements.txt").read_text(), re.M)
    return m.group(1) if m else None


def preflight(a: argparse.Namespace) -> None:
    """Stop before any training if this machine cannot run the pilot.

    a: the parsed arguments of main(). Prints the Python, torch and transformers versions, and on a CUDA machine the
    GPU name and the first 1,500 characters of the nvidia-smi output. Exits with a STOP message when there is no CUDA
    GPU (unless a.allow_cpu) or when the installed transformers differs from the transformers==X pin in
    requirements.txt (without a pin there is no check). Unless a.skip_model_check, it also loads the tokenizer and
    checkpoint a.model (downloading them if they are not cached, and raising transformers' own error if that fails)
    and prints the model's size in millions of parameters. torch and transformers are imported in every case."""
    import torch
    import transformers
    print("Python:", sys.version.split()[0], "torch:", torch.__version__, "transformers:", transformers.__version__)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout[:1500])
    elif not a.allow_cpu:
        sys.exit("STOP: no CUDA GPU. Kaggle: Settings > Accelerator > GPU. Colab: Runtime > Change runtime type > "
                 "T4 GPU. Do not train on CPU.")
    want = required_transformers()
    if want and transformers.__version__ != want:
        sys.exit(f"STOP: transformers {transformers.__version__} is installed but {want} is pinned: "
                 f"pip install -r {ROOT / 'requirements.txt'}")
    if not a.skip_model_check:  # fail before any training if the checkpoint cannot be downloaded
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        AutoTokenizer.from_pretrained(a.model)
        m = T5ForConditionalGeneration.from_pretrained(a.model)
        print("MODEL READY:", a.model, round(sum(p.numel() for p in m.parameters()) / 1e6, 1), "million parameters")
        del m


def smoke_test(out: Path) -> None:
    """Tiny random model, word arm only: checks training and prediction end to end. Not a GPU memory test.

    Runs oevl.encoder_experiment --tiny --epochs 1 --arms word in a child process that writes into out (main() passes
    the sibling folder <out>_smoke, which is never packed or removed). Every other setting is encoder_experiment's
    default, so unlike the pilot arms this run uses --max-bytes 256 and leaves Zupitza's Grammar out (decision D3).
    The output is captured, and its last 3,000 characters are printed when the run ends. Exits with a STOP message if
    the run fails or does not write metrics.json and the predictions of both TESTS. A run longer than 15 minutes
    raises subprocess.TimeoutExpired."""
    cmd = [sys.executable, "-m", "oevl.encoder_experiment", "--tiny", "--epochs", "1", "--arms", "word",
           "--out", str(out)]
    p = subprocess.run(cmd, cwd=ROOT, env=env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                       timeout=900)
    print(p.stdout[-3000:])
    if p.returncode:
        sys.exit(f"STOP: smoke test failed ({p.returncode})")
    for f in ["metrics.json"] + [f"{t}_predictions.jsonl" for t in TESTS]:
        if not (out / f).is_file():
            sys.exit(f"STOP: smoke test wrote no {f}")
    print("SMOKE TEST PASS", flush=True)


def pack(base: Path) -> Path:
    """Zip every file under base into <base>.zip next to it and return the archive's path.

    Members are stored relative to base.parent, so their names start with base's folder name, which restore() relies
    on. The zip is written to <base>.zip.tmp and then renamed over <base>.zip, so a crash leaves the previous archive
    in place and never half a zip. Empty folders are not stored. Prints the archive's path and size in MB."""
    archive = base.with_name(base.name + ".zip")
    temp = archive.with_name(archive.name + ".tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for f in sorted(base.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(base.parent))
    temp.replace(archive)
    print("ARCHIVE READY:", archive, "MB:", round(archive.stat().st_size / 1e6, 2), flush=True)
    return archive


def restore(src: Path, base: Path) -> list[str]:
    """Copy finished arms (folders with COMPLETE.json) from an earlier <out>.zip or extracted <out>/ folder into base.

    src: a folder with the arm folders (P1 to P5) directly inside it, or a zip as pack() writes it. Any src that is
    not a folder is opened as a zip. A folder is read whatever its name, but a zip only under a top folder named
    exactly like base, so a zip made for another --out name restores nothing. Zip members with ".." or an absolute
    path are skipped, so nothing is written outside base. An arm already finished in base is kept, an unfinished copy
    in base is deleted and replaced, and an arm without COMPLETE.json in src is never copied. Nothing is validated
    here: main() validates each requested arm when it skips it. Prints what was restored and returns the copied arm
    folder names, sorted."""
    copied = []
    if src.is_dir():
        for arm in sorted(p.parent for p in src.glob("*/COMPLETE.json")):
            if not (base / arm.name / "COMPLETE.json").exists():
                shutil.rmtree(base / arm.name, ignore_errors=True)
                shutil.copytree(arm, base / arm.name)
                copied.append(arm.name)
    else:
        with zipfile.ZipFile(src) as z:
            names = [n for n in z.namelist() if ".." not in Path(n).parts and not Path(n).is_absolute()]
            finished = {Path(n).parts[1] for n in names
                        if Path(n).parts[0] == base.name and Path(n).name == "COMPLETE.json" and len(Path(n).parts) == 3}
            for arm in sorted(finished):
                if not (base / arm / "COMPLETE.json").exists():
                    shutil.rmtree(base / arm, ignore_errors=True)
                    for n in names:
                        if Path(n).parts[:2] == (base.name, arm):
                            z.extract(n, base.parent)
                    copied.append(arm)
    print(f"restored {', '.join(copied)} from {src}" if copied else f"nothing to restore from {src}")
    return copied


def validate(out: Path, arm: str, a: argparse.Namespace) -> str:
    """Check one arm's outputs in out against the run's settings and results/wikipron, and return its subset checksum.

    These are the checks the notebooks made after every arm. metrics.json must record the same epochs, max_bytes,
    seed, batch, lr and train_limit as a, exactly train_limit training examples (not checked with --tiny, whose
    subset is smaller) and results for arm. For each of TESTS, the predictions file must have no token id twice and
    an encoder_<arm> prediction on every line, its ids must equal those of results/wikipron (a subset with --tiny),
    and metrics.json's accuracy n must equal the number of lines. micro, model and exclude_train are not checked.
    Raises AssertionError on a failed check, and KeyError or FileNotFoundError when a key or file is missing.
    Returns metrics.json's train_subset_sha256, which main() requires to be the same for every arm."""
    d = json.loads((out / "metrics.json").read_text(encoding="utf8"))
    want = {"epochs": a.epochs, "max_bytes": a.max_bytes, "seed": a.seed, "batch": a.batch, "lr": a.lr,
            "train_limit": a.train_limit}
    got = {k: d.get(k) for k in want}
    if got != want:
        raise AssertionError(f"{out.name}: settings {got} != {want}")
    if not a.tiny and d["train_examples"] != a.train_limit:
        raise AssertionError(f"{out.name}: trained on {d['train_examples']} examples, not {a.train_limit}")
    if arm not in d["arms"]:
        raise AssertionError(f"{out.name}: arm {arm} missing from metrics.json")
    for test in TESTS:
        seen = set()
        with open(out / f"{test}_predictions.jsonl", encoding="utf8") as f:
            for line in f:
                x = json.loads(line)
                if x["id"] in seen or f"encoder_{arm}" not in x:
                    raise AssertionError(f"{out.name} {test}: duplicate id or no encoder_{arm} prediction: {x['id']}")
                seen.add(x["id"])
        with open(ROOT / "results" / "wikipron" / f"{test}_predictions.jsonl", encoding="utf8") as f:
            reference = {json.loads(line)["id"] for line in f}
        if not (seen <= reference if a.tiny else seen == reference):  # --tiny scores only the first 300 tokens
            raise AssertionError(f"{out.name} {test}: token ids differ from results/wikipron")
        if d["arms"][arm][test]["accuracy"]["n"] != len(seen):
            raise AssertionError(f"{out.name} {test}: metrics n differs from the number of predictions")
    return d["train_subset_sha256"]


def run_arm(out: Path, arm: str, a: argparse.Namespace) -> int:
    """Train and score one arm with oevl.encoder_experiment in a child process that writes its outputs into out.

    Passes the pilot settings from a, --exclude-train "" so the arm trains on all five works, and --tiny when set.
    The child's stdout and stderr are echoed live and written to out/train.log, which is overwritten. Returns the
    child's exit code."""
    cmd = [sys.executable, "-m", "oevl.encoder_experiment", "--model", a.model, "--arms", arm,
           "--train-limit", str(a.train_limit), "--epochs", str(a.epochs), "--max-bytes", str(a.max_bytes),
           "--batch", str(a.batch), "--micro", str(a.micro), "--lr", str(a.lr), "--seed", str(a.seed),
           "--exclude-train", "",  # the pilots ran on all five works, before decision D3 was decided
           "--out", str(out)] + (["--tiny"] if a.tiny else [])
    with open(out / "train.log", "w", encoding="utf8") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             bufsize=1)
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return p.wait()


def main() -> None:
    """Run the pilot from the command line: preflight, smoke test, restore, then each requested arm in order.

    An arm whose folder already holds COMPLETE.json (from an earlier run into this --out or from --restore) is
    validated and skipped. Every other arm is run, validated, marked COMPLETE.json and packed. When a new arm fails,
    fails a check in validate() (AssertionError) or trained on a different subset from the other arms, the results
    folder is packed and the run stops with an error. Any other error, including a skipped arm that fails validation,
    stops the run without packing. Arms run in --arms order with duplicates dropped. No new arm starts once
    --max-hours have passed since the arguments were parsed, a clock that includes preflight, the smoke test and the
    restore: the folder is then packed and the run ends normally (the notebooks pass the hours left of their 9-hour
    budget). Ends by printing each requested arm's accuracy on TESTS, or NOT COMPLETED."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="results folder; <out>.zip is written next to it")
    ap.add_argument("--max-bytes", type=int, required=True, help="input cap in bytes: 128 or 256")
    ap.add_argument("--arms", default=",".join(ARMS), help="comma list from " + ", ".join(ARMS))
    ap.add_argument("--train-limit", type=int, default=10000)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--model", default="google/byt5-small")
    ap.add_argument("--restore", action="append", default=[], help="earlier <out>.zip or <out>/ folder; repeatable")
    ap.add_argument("--max-hours", type=float, default=9.0, help="start no new arm after this many hours")
    ap.add_argument("--tiny", action="store_true", help="tiny random model on a small subset (CPU dry run)")
    ap.add_argument("--allow-cpu", action="store_true", help="do not stop when there is no GPU (dry runs only)")
    ap.add_argument("--skip-model-check", action="store_true", help="do not download the checkpoint in preflight")
    a = ap.parse_args()
    getattr(sys.stdout, "reconfigure", lambda **_: None)(line_buffering=True)  # notebooks read this through a pipe
    arms = list(dict.fromkeys(a.arms.split(",")))
    if [x for x in arms if x not in ARMS]:
        ap.error(f"--arms must come from {', '.join(ARMS)}")

    start = time.time()
    base = Path(a.out).resolve()
    base.mkdir(parents=True, exist_ok=True)
    preflight(a)
    smoke_test(base.with_name(base.name + "_smoke"))
    for src in a.restore:
        restore(Path(src), base)

    shas = set()
    for arm in arms:
        eid = ARMS[arm]
        out = base / eid
        out.mkdir(exist_ok=True)
        if (out / "COMPLETE.json").exists():
            shas.add(validate(out, arm, a))
            if len(shas) > 1:
                sys.exit(f"STOP: {eid} {arm} trained on a different subset from the other arms: {sorted(shas)}")
            print("SKIP completed", eid, arm, flush=True)
            continue
        if time.time() - start > a.max_hours * 3600:
            pack(base)
            print(f"SAFE STOP: no time left to start another arm (--max-hours {a.max_hours:.2f}). "
                  "Download the archive and resume in a new session with --restore.")
            break
        print("START", eid, arm, a.max_bytes, "bytes", a.epochs, "epoch(s)", flush=True)
        t0 = time.time()
        status = run_arm(out, arm, a)
        if status:
            pack(base)
            sys.exit(f"{eid} {arm} failed ({status}): the archive holds the finished arms; see {out / 'train.log'}")
        try:
            shas.add(validate(out, arm, a))
        except AssertionError:
            pack(base)  # keep the unvalidated outputs; without COMPLETE.json a resume reruns this arm
            raise
        if len(shas) > 1:
            pack(base)
            sys.exit(f"STOP: {eid} {arm} trained on a different subset from the other arms: {sorted(shas)}")
        (out / "COMPLETE.json").write_text(json.dumps({
            "id": eid, "arm": arm, "max_bytes": a.max_bytes, "train_limit": a.train_limit, "epochs": a.epochs,
            "seed": a.seed, "duration_seconds": round(time.time() - t0), "status": "completed"}, indent=2))
        pack(base)
        print("COMPLETE", eid, arm, "minutes", round((time.time() - t0) / 60, 1), flush=True)
    else:
        print(f"ALL {len(arms)} {a.max_bytes}-BYTE PILOT ARMS COMPLETED")

    for arm in arms:  # summary of what is finished
        out = base / ARMS[arm]
        if (out / "COMPLETE.json").exists():
            m = json.loads((out / "metrics.json").read_text(encoding="utf8"))["arms"][arm]
            print(ARMS[arm], arm, *(f"{t}: {m[t]['accuracy']['accuracy']}" for t in TESTS))
        else:
            print(ARMS[arm], arm, "NOT COMPLETED")


if __name__ == "__main__":
    main()
