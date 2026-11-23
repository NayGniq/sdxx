"""Paired error diagnosis on the baseline Fashion-MNIST validation split."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mindspore as ms
import numpy as np

from fashion_data import NAMES, ROOT, download, load, patches, select
from model import FashionViT
from train import configure, predict


CASES = {"full49": ("full", 49), "energy25": ("energy", 25),
         "checkerboard25": ("checkerboard", 25)}
SHIRT = NAMES.index("Shirt")
BAG = NAMES.index("Bag")
COLORS = ("#737680", "#c05526", "#24747c")


def save_rows(path, rows):
    rows = iter(rows)
    first = next(rows)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def measures(labels, predicted, full):
    correct = predicted == labels
    new_error = (full == labels) & ~correct
    to_bag = (predicted == BAG) & (labels != BAG)
    return dict(samples=len(labels), correct=int(correct.sum()), accuracy=correct.mean(),
                predicted_bag=int((predicted == BAG).sum()), non_bag=int((labels != BAG).sum()),
                false_bag=int(to_bag.sum()), full_correct=int((full == labels).sum()),
                full_wrong=int((full != labels).sum()), new_errors=int(new_error.sum()),
                corrected=int(((full != labels) & correct).sum()),
                new_errors_to_bag=int((new_error & to_bag).sum()))


def make_figure(out, summaries, quartiles):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    ax = axes[0]
    for i, (row, color) in enumerate(zip(summaries, COLORS)):
        pct = 100 * row["accuracy"]
        ax.bar(i, pct, color=color, width=0.65)
        ax.text(i, pct + 2, f"{pct:.1f}%", ha="center")
    ax.set(title="All 6,000 validation images", ylabel="Accuracy (%)", ylim=(0, 100),
           xticks=range(3), xticklabels=["Full 49", "Energy 25", "Checker 25"])
    ax = axes[1]
    for name, color in zip(CASES, COLORS):
        rows = [r for r in quartiles if r["condition"] == name]
        ax.plot(range(1, 5), [100 * r["false_bag"] / r["samples"] for r in rows],
                marker="o", color=color, label=name)
    ax.set(title="600 true shirts, grouped by mean intensity",
           ylabel="Shirts classified as Bag (%)", xlabel="Brightness quartile (dark to light)",
           xticks=range(1, 5), yticks=range(0, 101, 20), ylim=(-2, 103))
    ax.legend()
    fig.savefig(out / "diagnosis.png", dpi=160)
    plt.close(fig)


def make_examples(out, images, labels, raw_ids, quartile, predictions):
    full, energy, checker = (predictions[key] for key in CASES)
    matches = (labels == SHIRT) & (full == SHIRT) & (energy == BAG) & (checker == SHIRT)
    examples = []
    for q in range(1, 5):
        valid = np.flatnonzero(matches & (quartile == q))
        if valid.size:
            idx = valid[np.argmin(raw_ids[valid])]
            examples.append(dict(quartile=q, eligible=len(valid), val_row=int(idx),
                                 raw_train_index=int(raw_ids[idx])))
    if not examples:
        raise RuntimeError("No validation shirts meet the illustrative example criteria.")
    save_rows(out / "example_indices.csv", examples)
    fig, axes = plt.subplots(len(examples), len(CASES), figsize=(8, 2.35 * len(examples)),
                             squeeze=False, layout="constrained")
    for axes_row, record in zip(axes, examples):
        i = record["val_row"]
        for ax, (name, (mode, k)) in zip(axes_row, CASES.items()):
            _, ids = select(patches(images[i:i+1]), mode, k)
            keep = np.zeros(49, dtype=bool)
            keep[ids[0]] = True
            keep = keep.reshape(7, 7).repeat(4, axis=0).repeat(4, axis=1)
            display = np.repeat(images[i, :, :, None], 3, axis=-1)
            display[~keep] = [136, 176, 191]
            ax.imshow(display, interpolation="nearest")
            ax.set(xticks=[], yticks=[], title=f"{name} → {NAMES[predictions[name][i]]}")
        axes_row[0].set_ylabel(f"Q{record['quartile']}  #{raw_ids[i]}")
    fig.suptitle("True class: Shirt | Raw training indices | Blue: omitted patches")
    fig.savefig(out / "examples.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "01_diagnosis")
    args = parser.parse_args()
    configure(args.threads)
    download()
    baseline = ROOT / "results"
    model = FashionViT()
    missing, extra = ms.load_param_into_net(model, ms.load_checkpoint(str(baseline / "best.ckpt")),
                                             strict_load=True)
    if missing or extra:
        raise ValueError(f"Checkpoint mismatch: {missing}, {extra}")
    model.set_train(False)
    with np.load(baseline / "split.npz") as split:
        raw_ids = split["validation"]
    raw_images, raw_labels = load("train")
    images, labels = raw_images[raw_ids], raw_labels[raw_ids]
    if len(np.unique(raw_ids)) != 6000 or not np.all(np.bincount(labels, minlength=10) == 600):
        raise ValueError("Expected 600 unique validation images per class.")
    predictions = {}
    for name, (mode, k) in CASES.items():
        predictions[name] = predict(model, images, mode, k)
        print(name, int((predictions[name] == labels).sum()), "/", len(labels), flush=True)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    brightness = images.mean(axis=(1, 2))
    shirt = labels == SHIRT
    cuts = np.quantile(brightness[shirt], [0.25, 0.5, 0.75])
    quartile = np.zeros(len(labels), dtype=int)
    quartile[shirt] = 1 + np.searchsorted(cuts, brightness[shirt], side="left")
    full = predictions["full49"]
    summaries, by_class, by_quartile = [], [], []
    for name, pred in predictions.items():
        summaries.append(dict(condition=name, **measures(labels, pred, full)))
        for label, category in enumerate(NAMES):
            mask = labels == label
            by_class.append(dict(condition=name, label=label, category=category,
                                 **measures(labels[mask], pred[mask], full[mask])))
        for q in range(1, 5):
            mask = quartile == q
            by_quartile.append(dict(condition=name, quartile=q, lower_exclusive=cuts[q-2] if q > 1 else "",
                                    upper_inclusive=cuts[q-1] if q < 4 else "",
                                    min_observed=brightness[mask].min(), max_observed=brightness[mask].max(),
                                    **measures(labels[mask], pred[mask], full[mask])))
    save_rows(out / "summary.csv", summaries)
    save_rows(out / "per_class.csv", by_class)
    save_rows(out / "shirt_quartiles.csv", by_quartile)
    save_rows(out / "validation_predictions.csv", (
        dict(val_row=i, raw_train_index=rid, actual=labels[i], mean_brightness=brightness[i],
             shirt_quartile=quartile[i] if shirt[i] else "",
             **{name: pred[i] for name, pred in predictions.items()}) for i, rid in enumerate(raw_ids)))
    make_figure(out, summaries, by_quartile)
    make_examples(out, images, labels, raw_ids, quartile, predictions)


if __name__ == "__main__":
    main()
