"""Train on full inputs and select one checkpoint using validation accuracy."""
import argparse
import csv
import json
import time
from pathlib import Path

import mindspore as ms
import numpy as np
from mindspore import nn

from fashion_data import ROOT, download, load, patches, select, split
from model import FashionViT


def inputs(images, mode="full", k=49, rng=None):
    blocks, ids = select(patches(images), mode, k, rng)
    return ms.Tensor(blocks), ms.Tensor(ids)


def predict(model, images, mode="full", k=49, seed=0, batch=64):
    rng = np.random.default_rng(seed)
    outputs = []
    for start in range(0, len(images), batch):
        args = inputs(images[start:start + batch], mode, k, rng)
        outputs.append(model(*args).asnumpy().argmax(axis=1))
    return np.concatenate(outputs)


def configure(threads):
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU", runtime_num_threads=threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--per-class", type=int, default=1000)
    parser.add_argument("--rates", type=float, nargs="+", default=[0.001, 0.0003])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1 or min(args.rates) <= 0:
        parser.error("epochs, threads and rates must be positive")
    configure(args.threads)
    download()
    images, labels = load("train")
    train, val = split(labels, args.per_class, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    if (args.out / "history.csv").exists():
        parser.error("该目录已有训练记录，请用 --out 指定新的实验目录")
    np.savez(args.out / "split.npz", train=train, validation=val)
    config = dict(vars(args), out=str(args.out), mindspore=ms.__version__, batch=64)
    (args.out / "config.json").write_text(json.dumps(config, indent=2))
    best = -1
    with (args.out / "history.csv").open("w", newline="") as file:
        history = csv.writer(file)
        history.writerow(["learning_rate", "epoch", "train_loss", "validation_accuracy", "seconds"])
        for rate in args.rates:
            ms.set_seed(args.seed)
            rng = np.random.default_rng(args.seed)
            model = FashionViT()
            optimizer = nn.AdamWeightDecay(model.trainable_params(), learning_rate=rate, weight_decay=0.01)
            criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

            def loss_fn(blocks, ids, targets):
                return criterion(model(blocks, ids), targets)

            gradient = ms.value_and_grad(loss_fn, None, optimizer.parameters)

            @ms.jit
            def step(blocks, ids, targets):
                loss, grads = gradient(blocks, ids, targets)
                return ms.ops.depend(loss, optimizer(grads))

            for epoch in range(1, args.epochs + 1):
                model.set_train(True)
                start = time.perf_counter()
                total_loss = 0.0
                order = rng.permutation(train)
                for offset in range(0, len(order), 64):
                    indices = order[offset:offset + 64]
                    loss = step(*inputs(images[indices]), ms.Tensor(labels[indices]))
                    total_loss += float(loss.asnumpy()) * len(indices)
                model.set_train(False)
                accuracy = float(np.mean(predict(model, images[val]) == labels[val]))
                row = [rate, epoch, total_loss / len(order), accuracy, time.perf_counter() - start]
                history.writerow(row)
                file.flush()
                print("lr=%g epoch=%d loss=%.4f val=%.4f seconds=%.1f" % tuple(row), flush=True)
                if accuracy > best:
                    best = accuracy
                    temp = args.out / "staging.ckpt"
                    ms.save_checkpoint(model, str(temp))
                    temp.replace(args.out / "best.ckpt")
                    winner = dict(rate=rate, epoch=epoch, validation_accuracy=accuracy)
                    (args.out / "best.json").write_text(json.dumps(winner, indent=2))


if __name__ == "__main__":
    main()
