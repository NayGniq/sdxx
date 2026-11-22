"""Fashion-MNIST 下载、分层划分和保留空间位置的选块。"""
import gzip
import hashlib
import shutil
import struct
import tempfile
from pathlib import Path
from urllib.request import urlopen

import numpy as np

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
CHECKSUMS = {
    "train-images-idx3-ubyte.gz": "8d4fb7e6c68d591d4c3dfef9ec88bf0d",
    "train-labels-idx1-ubyte.gz": "25c81989df183df01b3e8a0aad5dffbe",
    "t10k-images-idx3-ubyte.gz": "bef4ecab320f06d8554ea6380940ec79",
    "t10k-labels-idx1-ubyte.gz": "bb300cfdad3c16e7a12a480ee83cd310",
}
MIRRORS = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/",
    "https://fashion-mnist.s3.eu-central-1.amazonaws.com/",
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
)
NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal",
         "Shirt", "Sneaker", "Bag", "Ankle boot"]


def valid(path, checksum):
    return path.is_file() and hashlib.md5(path.read_bytes()).hexdigest() == checksum


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, checksum in CHECKSUMS.items():
        dest = RAW / name
        if valid(dest, checksum):
            continue
        errors = []
        for base in MIRRORS:
            with tempfile.NamedTemporaryFile(dir=RAW, delete=False) as temp:
                stage = Path(temp.name)
            try:
                with urlopen(base + name, timeout=45) as response, stage.open("wb") as out:
                    shutil.copyfileobj(response, out)
                if not valid(stage, checksum):
                    raise ValueError("MD5 mismatch")
                stage.replace(dest)
                break
            except (OSError, ValueError) as error:
                errors.append(f"{base}: {error}")
            finally:
                stage.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"{name} 下载失败: {errors}")
    print("Fashion-MNIST 四个文件均通过官方 MD5 校验。", flush=True)


def load(prefix):
    count = {"train": 60000, "t10k": 10000}[prefix]
    with gzip.open(RAW / f"{prefix}-images-idx3-ubyte.gz") as stream:
        if struct.unpack(">4I", stream.read(16)) != (2051, count, 28, 28):
            raise ValueError("Invalid image header")
        images = np.frombuffer(stream.read(), np.uint8).reshape(count, 28, 28)
    with gzip.open(RAW / f"{prefix}-labels-idx1-ubyte.gz") as stream:
        if struct.unpack(">2I", stream.read(8)) != (2049, count):
            raise ValueError("Invalid label header")
        labels = np.frombuffer(stream.read(), np.uint8).reshape(count).astype(np.int32)
    return images, labels


def split(labels, per_class, seed):
    if not 1 <= per_class <= 5400:
        raise ValueError("per-class must be in 1..5400")
    rng = np.random.default_rng(seed)
    train, validation = [], []
    for label in range(10):
        indices = rng.permutation(np.flatnonzero(labels == label))
        validation.extend(indices[:600])
        train.extend(indices[600:600 + per_class])
    return rng.permutation(train), np.sort(validation)


def patches(images):
    blocks = images.reshape(-1, 7, 4, 7, 4).transpose(0, 1, 3, 2, 4)
    return blocks.reshape(-1, 49, 16).astype(np.float32) / 255


def select(blocks, mode="full", k=49, rng=None):
    batch = len(blocks)
    if mode == "full":
        ids = np.broadcast_to(np.arange(49), (batch, 49))
    elif mode == "center":
        ids = np.broadcast_to(np.arange(49).reshape(7, 7)[1:6, 1:6].ravel(), (batch, 25))
    else:
        if mode == "energy":
            scores = np.square(blocks).sum(axis=2)
        elif mode == "random":
            scores = rng.random((batch, 49))
        else:
            raise ValueError(mode)
        # Equal energies use raster order. Keep the original position of every patch.
        ids = np.sort(np.argsort(-scores, kind="stable", axis=1)[:, :k], axis=1)
    chosen = blocks if mode == "full" else np.take_along_axis(blocks, ids[:, :, None], axis=1)
    return chosen, ids.astype(np.int32)


if __name__ == "__main__":
    download()
