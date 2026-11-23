"""Measure batch-one CPU latency and fixed-checkpoint accuracy for eight layouts."""
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mindspore as ms
import numpy as np

from evaluate import benchmark, write_rows
from fashion_data import ROOT, download, load
from model import FashionViT
from train import configure, predict


CASES = [
    ("full", 49), ("energy", 25), ("checkerboard", 25), ("region_energy", 25),
    ("center", 25), ("region_random", 25), ("energy", 35), ("random", 25),
]
OFFSETS = (0, 3, 6, 1, 4)
THREADS = 4
IMAGE_SEED = 2026
MASK_SEED = 17
DEFAULT_OUT = ROOT / "team_experiments" / "results" / "04_performance"


def condition(mode, k):
    return f"{mode}_{k}"


def load_model(checkpoint):
    configure(THREADS)
    model = FashionViT()
    missing, unused = ms.load_param_into_net(model, ms.load_checkpoint(str(checkpoint)),
                                             strict_load=True)
    if missing or unused:
        raise ValueError(f"Checkpoint mismatch: {missing}, {unused}")
    model.set_train(False)
    return model


def measure_round(checkpoint, out, round_no, cases):
    model = load_model(checkpoint)
    images, _ = load("t10k")
    indices = np.random.default_rng(IMAGE_SEED).choice(len(images), 300, replace=False)
    sample = images[indices]
    offset = OFFSETS[round_no - 1] % len(cases)
    rows = []
    for position, (mode, k) in enumerate(cases[offset:] + cases[:offset], start=1):
        times = benchmark(model, sample, mode, k, MASK_SEED)
        for image_index, (total, prepare, inference) in zip(indices, times):
            rows.append(dict(round=round_no, position=position, condition=condition(mode, k),
                             test_index=int(image_index), prepare_ms=prepare,
                             inference_ms=inference, total_ms=total))
    write_rows(out / f"round-{round_no}.csv", rows)
    print(f"Timing round {round_no} finished ({len(rows)} observations)", flush=True)


def plot_results(summary, path):
    labels = [row["condition"].replace("_", " ") for row in summary]
    y = np.arange(len(labels))
    fig, (left, right) = plt.subplots(1, 2, sharey=True, figsize=(10, 4.9),
                                      gridspec_kw={"width_ratios": [1.35, 1]},
                                      layout="constrained")
    p50 = [row["p50_ms"] for row in summary]
    p95 = [row["p95_ms"] for row in summary]
    left.hlines(y, p50, p95, color="#96a6b4", lw=2)
    left.scatter(p50, y, color="#215878", label="P50", zorder=2)
    left.scatter(p95, y, color="#d86a33", marker="D", label="P95", zorder=2)
    left.set(yticks=y, yticklabels=labels, xlabel="End-to-end CPU latency (ms)", xlim=(0, None))
    left.invert_yaxis()
    left.legend(loc="lower left")
    acc = [100 * row["test_accuracy"] for row in summary]
    right.barh(y, acc, height=0.55, color="#3d7971")
    for pos, value in zip(y, acc):
        right.text(value + 1, pos, f"{value:.2f}%", va="center", fontsize=9)
    right.set(xlabel="Official test accuracy (%)", xlim=(0, 105))
    for ax in (left, right):
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Fixed checkpoint  |  4 CPU threads  |  latency: batch 1, accuracy: batch 64")
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-pair", action="store_true",
                        help="Benchmark only full and checkerboard on the mixed-training checkpoint")
    parser.add_argument("--checkpoint", type=Path, help="Override the default checkpoint")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--worker-round", type=int, choices=range(1, 6), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mixed_pair:
        cases = [("full", 49), ("checkerboard", 25)]
        default_checkpoint = DEFAULT_OUT.parent / "03_training" / "best_mixed.ckpt"
        default_out = DEFAULT_OUT / "mixed_pair"
    else:
        cases = CASES
        default_checkpoint = ROOT / "results" / "best.ckpt"
        default_out = DEFAULT_OUT
    out = (args.out or default_out).resolve()
    checkpoint = (args.checkpoint or default_checkpoint).resolve()
    if args.worker_round:
        measure_round(checkpoint, out, args.worker_round, cases)
        return
    download()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS=str(THREADS),
               MKL_NUM_THREADS=str(THREADS))
    raw = []
    with tempfile.TemporaryDirectory(dir=out) as folder:
        for round_no in range(1, len(OFFSETS) + 1):
            subprocess.run([sys.executable, "-m", "team_experiments.performance",
                            "--checkpoint", str(checkpoint), "--out", folder,
                            "--worker-round", str(round_no)] + (["--mixed-pair"] if args.mixed_pair else []),
                           cwd=ROOT, env=env, check=True)
            with (Path(folder) / f"round-{round_no}.csv").open(newline="") as stream:
                raw.extend(csv.DictReader(stream))
    write_rows(out / "timings.csv", raw)

    rounds = []
    model = load_model(checkpoint)
    images, labels = load("t10k")
    accuracies = {}
    for mode, k in cases:
        name = condition(mode, k)
        predicted = predict(model, images, mode, k, MASK_SEED)
        accuracies[name] = int(np.count_nonzero(predicted == labels))
        for round_no in range(1, len(OFFSETS) + 1):
            block = [r for r in raw if r["condition"] == name and int(r["round"]) == round_no]
            if len({r["test_index"] for r in block}) != 300 or len(block) != 300:
                raise ValueError(f"Incomplete round {round_no} for {name}")
            total = [float(r["total_ms"]) for r in block]
            rounds.append(dict(condition=name, round=round_no, position=int(block[0]["position"]),
                               p50_ms=float(np.median(total)), p95_ms=float(np.percentile(total, 95)),
                               prepare_p50_ms=float(np.median([float(r["prepare_ms"]) for r in block])),
                               inference_p50_ms=float(np.median([float(r["inference_ms"]) for r in block]))))
    full_by_round = {r["round"]: r["p50_ms"] for r in rounds if r["condition"] == "full_49"}
    for r in rounds:
        r["speedup_vs_full"] = full_by_round[r["round"]] / r["p50_ms"]
    write_rows(out / "rounds.csv", rounds)
    summary = []
    for name, correct in accuracies.items():
        group = [r for r in rounds if r["condition"] == name]
        row = dict(condition=name, test_correct=correct, test_total=len(labels),
                   test_accuracy=correct / len(labels))
        for key in ("p50_ms", "p95_ms", "prepare_p50_ms", "inference_p50_ms", "speedup_vs_full"):
            row[key] = float(np.median([r[key] for r in group]))
        ratios = [r["speedup_vs_full"] for r in group]
        row.update(speedup_min=min(ratios), speedup_max=max(ratios),
                   faster_rounds=sum(value > 1 for value in ratios))
        summary.append(row)
    write_rows(out / "summary.csv", summary)
    plot_results(summary, out / "accuracy_latency.png")
    metadata = dict(platform=platform.platform(), processor=platform.processor(),
                    python=platform.python_version(), mindspore=ms.__version__, numpy=np.__version__,
                    checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    device="CPU", threads=THREADS, timing_batch=1, accuracy_batch=64,
                    images_per_round=300,
                    warmups_per_condition=30, independent_processes=len(OFFSETS),
                    image_seed=IMAGE_SEED, mask_seed=MASK_SEED, case_order=cases,
                    rotation_offsets=[offset % len(cases) for offset in OFFSETS],
                    timed_source="official test set; original indices in timings.csv",
                    aggregation="median over 5 round-specific statistics; speedups paired by round")
    (out / "environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
