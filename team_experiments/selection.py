"""Evaluate spatially constrained 25-patch selection on the frozen reference ViT."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mindspore as ms
import numpy as np

from fashion_data import NAMES, REGION_CANDIDATES, REGION_EDGES, ROOT, download, load, patches, select
from model import FashionViT
from train import configure, predict


OUT = ROOT / "team_experiments" / "results" / "02_selection"
RANDOM_SEEDS = (17, 29, 43)
MODES = ("full", "random", "center", "energy", "checkerboard", "region_random", "region_energy")
RULE = ("Among 25-patch methods, rank by mean validation accuracy descending, then by mean "
        "validation macro F1 descending, then by mode name ascending. Random policies use "
        "the unweighted mean of three seed-specific metrics.")


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def check_regions():
    """Check geometry, deterministic ties, original positions and RNG batching."""
    blocks = np.random.default_rng(501).random((23, 49, 16)).astype(np.float32)
    regions = [np.unique(region) for region in REGION_CANDIDATES]
    assert np.array_equal(np.sort(np.concatenate(regions)), np.arange(49))
    for mode in ("region_random", "region_energy"):
        chosen, ids = select(blocks, mode, 25, np.random.default_rng(17))
        rng = np.random.default_rng(17)
        chunked = np.concatenate([select(part, mode, 25, rng)[1] for part in np.array_split(blocks, 4)])
        assert np.array_equal(ids, chunked) and np.all(np.diff(ids, axis=1) > 0)
        assert np.array_equal(chosen, np.take_along_axis(blocks, ids[:, :, None], axis=1))
        assert all(np.all(np.isin(ids, region).sum(axis=1) == 1) for region in regions)
    _, ties = select(np.zeros((1, 49, 16), dtype=np.float32), "region_energy", 25)
    assert np.array_equal(ties[0], np.sort([region.min() for region in regions]))


def score(labels, predicted):
    confusion = np.bincount(labels * 10 + predicted, minlength=100).reshape(10, 10)
    f1 = 2 * confusion.diagonal() / np.maximum(1, confusion.sum(axis=0) + confusion.sum(axis=1))
    return float(np.mean(predicted == labels)), float(f1.mean())


def evaluate_split(model, images, labels, indices, name, out):
    rows, summary = [], []
    predictions = {"actual": labels, "indices": indices}
    for mode in MODES:
        k = 49 if mode == "full" else 25
        seeds = RANDOM_SEEDS if mode in ("random", "region_random") else (0,)
        runs = []
        for seed in seeds:
            predicted = predict(model, images, mode, k, seed)
            accuracy, macro_f1 = score(labels, predicted)
            key = f"{mode}{k}" + (f"_s{seed}" if len(seeds) > 1 else "")
            predictions[key] = predicted.astype(np.uint8)
            row = dict(mode=mode, k=k, seed=seed if len(seeds) > 1 else "", n=len(labels),
                       correct=int(np.sum(predicted == labels)), accuracy=accuracy, macro_f1=macro_f1)
            rows.append(row)
            runs.append(row)
        grouped = dict(mode=mode, k=k, runs=len(runs))
        for metric in ("accuracy", "macro_f1"):
            values = [row[metric] for row in runs]
            grouped[metric] = float(np.mean(values))
            grouped[metric + "_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        summary.append(grouped)
        print(f"{name} {mode}: accuracy={grouped['accuracy']:.5f} macro_f1={grouped['macro_f1']:.5f}", flush=True)
    write_csv(out / f"{name}_runs.csv", rows)
    write_csv(out / f"{name}_summary.csv", summary)
    np.savez_compressed(out / f"{name}_predictions.npz", **predictions)
    return summary


def plot_selections(images, labels, out):
    # Predefined illustration: first official test image from each of these classes.
    illustration_ids = [int(np.flatnonzero(labels == cls)[0]) for cls in (0, 6, 9)]
    selected_modes = ("full", "energy", "checkerboard", "region_random", "region_energy")
    titles = ("Full (49)", "Global energy", "Checkerboard", "Regional random", "Regional energy")
    fig, axes = plt.subplots(len(illustration_ids), len(titles), figsize=(10, 6), squeeze=False)
    for row, index in enumerate(illustration_ids):
        for col, mode in enumerate(selected_modes):
            rng = np.random.default_rng(RANDOM_SEEDS[0])
            if mode == "region_random":
                rng.random((index, 49))
            _, ids = select(patches(images[index:index + 1]), mode, 49 if mode == "full" else 25, rng)
            keep = np.zeros(49, dtype=bool)
            keep[ids[0]] = True
            mask = keep.reshape(7, 7).repeat(4, axis=0).repeat(4, axis=1)
            display = np.repeat(images[index, :, :, None], 3, axis=2)
            display[~mask] = [84, 129, 159]
            ax = axes[row, col]
            ax.imshow(display, interpolation="nearest")
            if mode.startswith("region"):
                for edge in REGION_EDGES[1:-1]:
                    ax.axhline(4 * edge - 0.5, color="gold", linewidth=0.6)
                    ax.axvline(4 * edge - 0.5, color="gold", linewidth=0.6)
            ax.set(xticks=[], yticks=[])
            if row == 0:
                ax.set_title(titles[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(f"{NAMES[labels[index]]}\nimage {index}")
    fig.suptitle("Which patches reach the ViT? Blue = removed, yellow lines = regional boundaries")
    fig.tight_layout()
    fig.savefig(out / "selection_masks.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    out = args.out
    out.mkdir(exist_ok=True, parents=True)
    check_regions()
    configure(args.threads)
    download()
    reference = ROOT / "results"
    provenance = {name: hashlib.sha256((reference / name).read_bytes()).hexdigest()
                  for name in ("best.ckpt", "split.npz")}
    save_json(out / "protocol.json", dict(rule=RULE, modes=MODES, seeds=RANDOM_SEEDS,
                                          region_edges=REGION_EDGES, sha256=provenance,
                                          threads=args.threads, mindspore=ms.__version__, numpy=np.__version__))
    model = FashionViT()
    missing, unused = ms.load_param_into_net(model, ms.load_checkpoint(str(reference / "best.ckpt")),
                                              strict_load=True)
    if missing or unused:
        raise ValueError(f"Reference checkpoint mismatch: {missing}, {unused}")
    model.set_train(False)
    train_images, train_labels = load("train")
    with np.load(reference / "split.npz") as split:
        val_ids = split["validation"]
    if len(np.unique(val_ids)) != 6000 or not np.all(np.bincount(train_labels[val_ids]) == 600):
        raise ValueError("Unexpected reference validation split")
    validation = evaluate_split(model, train_images[val_ids], train_labels[val_ids], val_ids, "validation", out)
    candidates = [row for row in validation if row["k"] == 25]
    winner = min(candidates, key=lambda row: (-row["accuracy"], -row["macro_f1"], row["mode"]))
    save_json(out / "selection.json", dict(rule=RULE, selected=winner))
    print(f"Frozen validation winner: {winner['mode']}", flush=True)
    test_images, test_labels = load("t10k")
    evaluate_split(model, test_images, test_labels, np.arange(len(test_labels)), "test", out)
    plot_selections(test_images, test_labels, out)


if __name__ == "__main__":
    main()
