"""Compare full-input and mixed-input training from an identical initial model."""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import mindspore as ms
import numpy as np

from fashion_data import ROOT, download, load
from model import FashionViT
from train import configure, make_training_step, predict, save_best_checkpoint, train_epoch


EVALUATION_INPUTS = (("full", 49), ("checkerboard", 25))
DEFAULT_OUTPUT = ROOT / "team_experiments" / "results" / "03_training"


def parameter_digest(model):
    digest = hashlib.sha256()
    for name, param in sorted(model.parameters_and_names()):
        digest.update(name.encode())
        digest.update(param.asnumpy().tobytes())
    return digest.hexdigest()


def restore(path):
    model = FashionViT()
    missing, unused = ms.load_param_into_net(model, ms.load_checkpoint(str(path)), strict_load=True)
    if missing or unused:
        raise ValueError(f"Checkpoint mismatch {missing}, {unused}")
    model.set_train(False)
    return model


def evaluate(model, images, labels):
    rows = []
    for mode, k in EVALUATION_INPUTS:
        predictions = predict(model, images, mode, k)
        matrix = np.bincount(labels * 10 + predictions, minlength=100).reshape(10, 10)
        f1 = 2 * matrix.diagonal() / np.maximum(1, matrix.sum(axis=0) + matrix.sum(axis=1))
        rows.append(dict(input=mode, patches=k, n=len(labels),
                         accuracy=float(np.mean(predictions == labels)), macro_f1=float(f1.mean())))
    return rows


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def train_method(args, mode, images, labels, train_ids, validation, initial_digest):
    ms.set_seed(args.seed)
    model = restore(args.out / "initial.ckpt")
    start_digest = parameter_digest(model)
    if start_digest != initial_digest:
        raise ValueError("Training did not start from the saved initialization")
    step = make_training_step(model, args.rate)
    rng = np.random.default_rng(args.seed)
    rows = []
    best_score = -1
    best_epoch = 0
    with (args.out / f"history_{mode}.csv").open("w", newline="") as stream:
        for epoch in range(1, args.epochs + 1):
            order = rng.permutation(train_ids)
            loss, full, checker = train_epoch(model, step, images, labels, order,
                                              batch=args.batch, mode=mode, epoch=epoch)
            val_full, val_checker = evaluate(model, *validation)
            score = (val_full["accuracy"] + val_checker["accuracy"]) / 2
            row = dict(epoch=epoch, loss=loss, full_val_acc=val_full["accuracy"],
                       checker_val_acc=val_checker["accuracy"], full_val_macro_f1=val_full["macro_f1"],
                       checker_val_macro_f1=val_checker["macro_f1"], selection_score=score,
                       full_batches=full, checker_batches=checker, optimizer_steps=full + checker,
                       order_sha256=hashlib.sha256(order.astype("<i8").tobytes()).hexdigest())
            if epoch == 1:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            rows.append(row)
            if score > best_score:
                best_score, best_epoch = score, epoch
                save_best_checkpoint(model, args.out / f"best_{mode}.ckpt")
            print(f"{mode} {epoch}/{args.epochs} loss={loss:.4f} "
                  f"val_full={val_full['accuracy']:.4f} val_checker={val_checker['accuracy']:.4f}", flush=True)
    audit = dict(mode=mode, initial_parameter_sha256=start_digest, best_epoch=best_epoch,
                 best_score=best_score, full_batches=sum(r["full_batches"] for r in rows),
                 checker_batches=sum(r["checker_batches"] for r in rows),
                 optimizer_steps=sum(r["optimizer_steps"] for r in rows))
    return rows, audit


def plot_curves(history, out):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(sys.prefix) / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for mode, rows in history.items():
        epochs = [r["epoch"] for r in rows]
        axes[0].plot(epochs, [r["loss"] for r in rows], marker=".", label=f"{mode} training")
        for key, label, style in [("full_val_acc", "49", "-"), ("checker_val_acc", "25", "--")]:
            axes[1].plot(epochs, [r[key] for r in rows], style, marker=".", label=f"{mode} / {label}")
    axes[0].set(xlabel="Epoch", ylabel="Mean training loss")
    axes[1].set(xlabel="Epoch", ylabel="Validation accuracy")
    for ax in axes:
        ax.legend()
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", choices=("full", "mixed"), default=["full", "mixed"])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--rate", type=float, default=0.0003)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--split", type=Path, default=ROOT / "results" / "split.npz")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if min(args.epochs, args.rate, args.batch, args.threads) <= 0 or len(set(args.modes)) != len(args.modes):
        parser.error("epochs, rate, batch and threads must be positive; modes must be unique")
    if (args.out / "config.json").exists():
        parser.error("请选择新的 --out 目录以保留已有结果")
    args.out.mkdir(parents=True, exist_ok=True)
    configure(args.threads)
    download()
    images, labels = load("train")
    with np.load(args.split) as split:
        train_ids, val_ids = split["train"], split["validation"]
    ids = np.concatenate([train_ids, val_ids])
    if len(np.unique(ids)) != len(ids) or ids.min() < 0 or ids.max() >= len(labels):
        raise ValueError("Split has duplicate, overlapping, or out-of-range IDs")
    validation = images[val_ids], labels[val_ids]
    ms.set_seed(args.seed)
    model = FashionViT()
    model.init_parameters_data()
    initial_digest = parameter_digest(model)
    ms.save_checkpoint(model, str(args.out / "initial.ckpt"))
    write_json(args.out / "config.json", dict(modes=args.modes, epochs=args.epochs, rate=args.rate,
               optimizer="AdamWeightDecay", weight_decay=0.01, batch=args.batch, seed=args.seed,
               device="CPU", threads=args.threads, mindspore=ms.__version__, numpy=np.__version__,
               split=os.path.relpath(args.split, ROOT), split_sha256=hashlib.sha256(args.split.read_bytes()).hexdigest(),
               train_n=len(train_ids), validation_n=len(val_ids), initial_parameter_sha256=initial_digest,
               selection="max mean validation accuracy on full49 and checkerboard25; earliest tie"))
    history, audits = {}, {}
    for mode in args.modes:
        history[mode], audits[mode] = train_method(args, mode, images, labels, train_ids, validation, initial_digest)
    if len(history) == 2:
        for left, right in zip(history["full"], history["mixed"]):
            if (left["order_sha256"], left["optimizer_steps"]) != (right["order_sha256"], right["optimizer_steps"]):
                raise ValueError("Mismatched order or optimizer step count")
    write_json(args.out / "audit.json", audits)
    test = load("t10k")
    with (args.out / "evaluation.csv").open("w", newline="") as stream:
        fields = ["training_mode", "best_epoch", "split", "input", "patches", "n", "accuracy", "macro_f1"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for mode in args.modes:
            model = restore(args.out / f"best_{mode}.ckpt")
            for split_name, dataset in [("validation", validation), ("test", test)]:
                rows = evaluate(model, *dataset)
                if split_name == "validation" and sum(r["accuracy"] for r in rows) / 2 != audits[mode]["best_score"]:
                    raise ValueError(f"Restored {mode} result differs from selected epoch")
                for row in rows:
                    writer.writerow(dict(training_mode=mode, best_epoch=audits[mode]["best_epoch"],
                                         split=split_name, **row))
    plot_curves(history, args.out)


if __name__ == "__main__":
    main()
