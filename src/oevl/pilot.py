"""Exploratory GPU pilot: the ByT5 encoder's context arms on a fixed training subset.

This is the logic that used to live in the four Kaggle/Colab notebooks. The notebooks in notebooks/ now only put
the project on the machine and call:

  python -m oevl.pilot --max-bytes 128 --out /kaggle/working/encoder_pilot_128_results [--restore OLD.zip]

Steps: check the GPU and the pinned transformers version, load the checkpoint, run a tiny CPU-sized smoke test,
then train and evaluate each arm in its own process. A finished arm is validated, marked COMPLETE.json and packed
into <out>.zip, so a later session resumes with --restore <out>.zip and skips it. There is no checkpoint inside
an arm: an arm that is interrupted starts again from the beginning.

Defaults are the agreed pilot: arms sentence, word, left, local2, local5 (P1-P5), 10,000 training examples,
1 epoch, batch 32, microbatch 8, learning rate 3e-4, seed 1. Only --max-bytes differs between the 128 and 256
pilots. CPU dry run of the whole loop: --tiny --allow-cpu --skip-model-check.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARMS = {"sentence": "P1", "word": "P2", "left": "P3", "local2": "P4", "local5": "P5"}
TESTS = ("soliloquies", "tauber_beowulf")


def env():
    e = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
             TOKENIZERS_PARALLELISM="false")
    e["PYTHONPATH"] = os.pathsep.join(x for x in (str(ROOT / "src"), e.get("PYTHONPATH")) if x)
    return e


def required_transformers():
    m = re.search(r"^transformers==(\S+)", (ROOT / "requirements-neural.txt").read_text(), re.M)
    return m.group(1) if m else None


def preflight(a):
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
                 f"pip install -r {ROOT / 'requirements-neural.txt'}")
    if not a.skip_model_check:  # fail before any training if the checkpoint cannot be downloaded
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        AutoTokenizer.from_pretrained(a.model)
        m = T5ForConditionalGeneration.from_pretrained(a.model)
        print("MODEL READY:", a.model, round(sum(p.numel() for p in m.parameters()) / 1e6, 1), "million parameters")
        del m


def smoke_test(out):
    """Tiny random model, word arm only: checks training and prediction end to end. Not a GPU memory test."""
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


def pack(base):
    """Zip everything under base into <base>.zip; written to a temp file first so a crash never leaves half a zip."""
    archive = base.with_name(base.name + ".zip")
    temp = archive.with_name(archive.name + ".tmp")
    with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED, compresslevel=4) as z:
        for f in sorted(base.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(base.parent))
    temp.replace(archive)
    print("ARCHIVE READY:", archive, "MB:", round(archive.stat().st_size / 1e6, 2), flush=True)
    return archive


def restore(archive, base):
    """Unpack an earlier <out>.zip into an empty <out>/; members outside <out>/ are refused."""
    if any(base.iterdir()):
        print("not restoring: output folder is not empty:", base)
        return
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            parts = Path(member).parts
            if parts and parts[0] == base.name and ".." not in parts and not Path(member).is_absolute():
                z.extract(member, base.parent)
    print("Restored previous results from", archive)


def validate(out, arm, a):
    """The checks the notebooks made after every arm, plus one subset checksum shared by all arms."""
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


def run_arm(out, arm, a):
    cmd = [sys.executable, "-m", "oevl.encoder_experiment", "--model", a.model, "--arms", arm,
           "--train-limit", str(a.train_limit), "--epochs", str(a.epochs), "--max-bytes", str(a.max_bytes),
           "--batch", str(a.batch), "--micro", str(a.micro), "--lr", str(a.lr), "--seed", str(a.seed),
           "--out", str(out)] + (["--tiny"] if a.tiny else [])
    with open(out / "train.log", "w", encoding="utf8") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             bufsize=1)
        for line in p.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return p.wait()


def main():
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
    ap.add_argument("--restore", help="earlier <out>.zip to resume from; ignored unless <out>/ is empty")
    ap.add_argument("--max-hours", type=float, default=9.0, help="start no new arm after this many hours")
    ap.add_argument("--tiny", action="store_true", help="tiny random model on a small subset (CPU dry run)")
    ap.add_argument("--allow-cpu", action="store_true", help="do not stop when there is no GPU (dry runs only)")
    ap.add_argument("--skip-model-check", action="store_true", help="do not download the checkpoint in preflight")
    a = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # the notebooks read this through a pipe; show lines as they come
    arms = a.arms.split(",")
    if [x for x in arms if x not in ARMS]:
        ap.error(f"--arms must come from {', '.join(ARMS)}")

    start = time.time()
    base = Path(a.out).resolve()
    base.mkdir(parents=True, exist_ok=True)
    preflight(a)
    smoke_test(base.with_name(base.name + "_smoke"))
    if a.restore:
        restore(Path(a.restore), base)

    shas = set()
    for arm in arms:
        eid = ARMS[arm]
        out = base / eid
        out.mkdir(exist_ok=True)
        if (out / "COMPLETE.json").exists():
            shas.add(validate(out, arm, a))
            print("SKIP completed", eid, arm, flush=True)
            continue
        if time.time() - start > a.max_hours * 3600:
            pack(base)
            print(f"SAFE STOP: over {a.max_hours} h since start. Download the archive and resume with --restore.")
            break
        print("START", eid, arm, a.max_bytes, "bytes", a.epochs, "epoch(s)", flush=True)
        t0 = time.time()
        status = run_arm(out, arm, a)
        if status:
            pack(base)
            sys.exit(f"{eid} {arm} failed ({status}): the archive holds the finished arms; see {out / 'train.log'}")
        shas.add(validate(out, arm, a))
        (out / "COMPLETE.json").write_text(json.dumps({
            "id": eid, "arm": arm, "max_bytes": a.max_bytes, "train_limit": a.train_limit, "epochs": a.epochs,
            "seed": a.seed, "duration_seconds": round(time.time() - t0), "status": "completed"}, indent=2))
        pack(base)
        print("COMPLETE", eid, arm, "minutes", round((time.time() - t0) / 60, 1), flush=True)
    else:
        print(f"ALL {len(arms)} {a.max_bytes}-BYTE PILOT ARMS COMPLETED")
    if len(shas) > 1:
        sys.exit(f"STOP: arms trained on different subsets: {sorted(shas)}")

    for arm in arms:  # summary of what is finished
        out = base / ARMS[arm]
        if (out / "COMPLETE.json").exists():
            m = json.loads((out / "metrics.json").read_text(encoding="utf8"))["arms"][arm]
            print(ARMS[arm], arm, *(f"{t}: {m[t]['accuracy']['accuracy']}" for t in TESTS))
        else:
            print(ARMS[arm], arm, "NOT COMPLETED")


if __name__ == "__main__":
    main()
