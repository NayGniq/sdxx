"""Compare fixed weights and benchmark batch-one CPU inference."""
import argparse
import csv
import json
import platform
from pathlib import Path
from time import perf_counter_ns

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mindspore as ms
import numpy as np

from fashion_data import NAMES, ROOT, download, load, patches, select
from model import FashionViT
from train import configure, inputs, predict

CASES = [("full", 49, 0), ("center", 25, 0), ("energy", 25, 0),
         ("energy", 35, 0)] + [("random", 25, seed) for seed in (17, 29, 43)]


def write_rows(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def benchmark(model, images, mode, k, seed):
    # Each condition warms its compiled batch-one shape. Copying logits to
    # NumPy synchronizes execution before the timer stops.
    rng = np.random.default_rng(seed)
    times = []
    for index in list(range(30)) + list(range(len(images))):
        image = images[index:index + 1]
        start = perf_counter_ns()
        tensors = inputs(image, mode, k, rng)
        prepared = perf_counter_ns()
        model(*tensors).asnumpy().argmax(axis=1)
        finished = perf_counter_ns()
        times.append(((finished - start) / 1e6, (prepared - start) / 1e6,
                      (finished - prepared) / 1e6))
    return np.asarray(times[30:])


def group_summary(rows):
    summary = []
    for mode, k in dict.fromkeys((row["mode"], row["k"]) for row in rows):
        group = [row for row in rows if (row["mode"], row["k"]) == (mode, k)]
        result = dict(mode=mode, k=k, runs=len(group))
        for metric in ("accuracy", "macro_f1", "median_ms"):
            values = [row[metric] for row in group]
            result[metric] = np.mean(values)
            result[metric + "_sd"] = np.std(values, ddof=1) if len(group) > 1 else 0
        summary.append(result)
    return summary


def plot_figures(out, images, labels, predictions, matrices):
    with (out / "history.csv").open() as file:
        history = list(csv.DictReader(file))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for rate in dict.fromkeys(row["learning_rate"] for row in history):
        rows = [row for row in history if row["learning_rate"] == rate]
        for ax, key in zip(axes, ["train_loss", "validation_accuracy"]):
            ax.plot([int(row["epoch"]) for row in rows], [float(row[key]) for row in rows],
                    marker=".", label=rate)
            ax.set(xlabel="Epoch", ylabel=key.replace("_", " ").title())
            ax.legend(title="Learning rate")
    fig.savefig(out / "training.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    for ax, key in zip(axes, ["full49", "energy25"]):
        ax.imshow(matrices[key], cmap="Blues", vmin=0, vmax=1000)
        ax.set(xticks=range(10), yticks=range(10), xticklabels=NAMES, yticklabels=NAMES,
               xlabel="Predicted", ylabel="Actual", title=key)
        ax.tick_params(axis="x", rotation=65)
        for (i, j), n in np.ndenumerate(matrices[key]):
            ax.text(j, i, str(n), ha="center", va="center", fontsize=7,
                    color="white" if n > 550 else "black")
    fig.savefig(out / "confusion.png", dpi=150)
    plt.close(fig)

    full = predictions["full49"]
    energy = predictions["energy25"]
    cases = np.flatnonzero((full == labels) & (energy != labels))
    shirts = cases[labels[cases] == 6]
    index = int(shirts[0] if len(shirts) else cases[0] if len(cases) else 0)
    fig, axes = plt.subplots(1, 5, figsize=(12, 3), constrained_layout=True)
    for ax, (mode, k, seed) in zip(axes, [CASES[0], CASES[4], CASES[1], CASES[2], CASES[3]]):
        # Consume the same per-image random stream as the stored predictions.
        batch = patches(images[index:index+1])
        rng = np.random.default_rng(seed)
        if mode == "random":
            rng.random((index, 49))
        _, ids = select(batch, mode, k, rng)
        kept = np.zeros(49, dtype=bool)
        kept[ids[0]] = True
        mask = kept.reshape(7, 7).repeat(4, axis=0).repeat(4, axis=1)
        display = np.repeat(images[index, :, :, None], 3, axis=2)
        display[~mask] = [116, 158, 185]
        key = f"{mode}{k}" + (f"_s{seed}" if mode == "random" else "")
        ax.imshow(display, interpolation="nearest")
        ax.set_title(f"{mode} {k}\n{NAMES[predictions[key][index]]}")
        ax.axis("off")
    fig.suptitle(f"Test image {index} | true: {NAMES[labels[index]]} | blue: removed")
    fig.savefig(out / "example.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--coverage-validation", action="store_true",
                        help="Compare 25-patch layouts on the validation split and write coverage_validation.csv")
    args = parser.parse_args()
    configure(args.threads)
    download()
    model = FashionViT()
    missing, unused = ms.load_param_into_net(model, ms.load_checkpoint(str(args.out / "best.ckpt")),
                                              strict_load=True)
    if missing or unused:
        raise ValueError(f"Checkpoint mismatch: {missing}, {unused}")
    model.set_train(False)
    # Both K values were specified in advance. Report validation results before
    # opening the official test split; test accuracy never chooses the weights.
    train_images, train_labels = load("train")
    with np.load(args.out / "split.npz") as saved:
        val_ids = saved["validation"]
    if args.coverage_validation:
        rows = []
        val_images, val_labels = train_images[val_ids], train_labels[val_ids]
        cases = [case for case in CASES if case[1] in (25, 49)] + [("checkerboard", 25, 0)]
        for mode, k, seed in cases:
            pred = predict(model, val_images, mode, k, seed)
            correct = int(np.count_nonzero(pred == val_labels))
            row = dict(mode=mode, k=k, seed=seed, correct=correct, total=len(pred),
                       accuracy=correct / len(pred))
            rows.append(row)
            print(row, flush=True)
        write_rows(args.out / "coverage_validation.csv", rows)
        return
    validation = []
    for mode, k in [("full", 49), ("energy", 25), ("energy", 35)]:
        pred = predict(model, train_images[val_ids], mode, k)
        validation.append(dict(mode=mode, k=k, accuracy=np.mean(pred == train_labels[val_ids])))
    write_rows(args.out / "validation.csv", validation)
    images, labels = load("t10k")
    speed_ids = np.random.default_rng(2026).choice(len(images), 300, replace=False)
    speed_images = images[speed_ids]
    predictions, matrices = {}, {}
    results, class_rows, timing_rows = [], [], []
    for mode, k, seed in CASES:
        key = f"{mode}{k}" + (f"_s{seed}" if mode == "random" else "")
        pred = predict(model, images, mode, k, seed)
        matrix = np.bincount(labels * 10 + pred, minlength=100).reshape(10, 10)
        actual, classified = matrix.sum(axis=1), matrix.sum(axis=0)
        f1 = 2 * matrix.diagonal() / np.maximum(1, actual + classified)
        times = benchmark(model, speed_images, mode, k, seed)
        predictions[key], matrices[key] = pred, matrix
        for i, name in enumerate(NAMES):
            class_rows.append(dict(condition=key, label=name, recall=matrix[i, i] / actual[i], f1=f1[i]))
        timing_rows.extend(dict(condition=key, test_index=i, ms=total,
                                prepare_ms=prepare, inference_ms=inference)
                           for i, (total, prepare, inference) in zip(speed_ids, times))
        row = dict(mode=mode, k=k, seed=seed, accuracy=float(np.mean(pred == labels)),
                   macro_f1=float(f1.mean()), median_ms=float(np.median(times[:, 0])))
        results.append(row)
        print(key, row, flush=True)

    write_rows(args.out / "test_runs.csv", results)
    write_rows(args.out / "summary.csv", group_summary(results))
    write_rows(args.out / "per_class.csv", class_rows)
    write_rows(args.out / "latency.csv", timing_rows)
    np.savez_compressed(args.out / "predictions.npz", actual=labels, **predictions)
    np.savez(args.out / "confusion.npz", **matrices)
    environment = dict(platform=platform.platform(), machine=platform.machine(), device="CPU",
                       threads=args.threads, mindspore=ms.__version__, numpy=np.__version__,
                       batch=1, warmup=30, timed_images=len(speed_ids))
    (args.out / "benchmark.json").write_text(json.dumps(environment, indent=2))
    plot_figures(args.out, images, labels, predictions, matrices)


if __name__ == "__main__":
    main()
