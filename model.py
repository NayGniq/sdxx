"""Two pre-norm Transformer blocks with learned absolute positions."""
import mindspore as ms
from mindspore import nn, ops
from mindspore.common.initializer import Normal, initializer


class Encoder(nn.Cell):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm((64,))
        self.qkv = nn.Dense(64, 192)
        self.out = nn.Dense(64, 64)
        self.norm2 = nn.LayerNorm((64,))
        self.mlp = nn.SequentialCell(nn.Dense(64, 128), nn.GELU(), nn.Dense(128, 64))

    def construct(self, x):
        batch, length, _ = x.shape
        qkv = self.qkv(self.norm1(x)).reshape(batch, length, 3, 4, 16)
        q, k, v = qkv.transpose(2, 0, 3, 1, 4)
        weights = ops.softmax(ops.matmul(q, k.swapaxes(-1, -2)) * 0.25, axis=-1)
        features = ops.matmul(weights, v).transpose(0, 2, 1, 3).reshape(batch, length, 64)
        x = x + self.out(features)
        return x + self.mlp(self.norm2(x))


class FashionViT(nn.Cell):
    def __init__(self):
        super().__init__()
        self.embed = nn.Dense(16, 64)
        self.position = ms.Parameter(initializer(Normal(0.02), (49, 64)), name="position")
        self.cls = ms.Parameter(initializer(Normal(0.02), (1, 1, 64)), name="cls")
        self.encoder = nn.SequentialCell(Encoder(), Encoder())
        self.norm = nn.LayerNorm((64,))
        self.head = nn.Dense(64, 10)

    def construct(self, blocks, ids):
        tokens = self.embed(blocks) + ops.gather(self.position, ids, 0)
        cls = ops.tile(self.cls, (blocks.shape[0], 1, 1))
        hidden = self.encoder(ops.concat((cls, tokens), axis=1))
        return self.head(self.norm(hidden[:, 0]))
