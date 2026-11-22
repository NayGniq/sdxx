# 基于视觉 Transformer 的服饰分类与图像块筛选

用同一组小型 ViT 权重，测量 Fashion-MNIST 图片只保留部分图块时的分类表现和 CPU 单图延迟。运行环境为 Python 3.9、MindSpore 2.6、NumPy 和 Matplotlib。当前项目可直接复用上级目录的虚拟环境。

## 本次实验结果

最佳权重取自学习率 0.0003 的第 14 轮，验证准确率为 84.13%。下表使用官方全部 10,000 张测试图，延迟取自本机 CPU 的 4 线程测量。

| 输入 | 测试准确率 | 宏平均 F1 | 单图耗时（毫秒） |
|---|---:|---:|---:|
| 完整 49 块 | 83.75% | 0.8375 | 0.482 |
| 随机 25 块（三个种子的均值） | 68.91% | 0.6876 | 0.406 |
| 中央 25 块 | 56.16% | 0.5170 | 0.398 |
| 能量最高 25 块 | 29.69% | 0.2753 | 0.403 |
| 能量最高 35 块 | 54.65% | 0.5498 | 0.418 |

能量选 25 块时，模型将 7,175 张测试图判为包，测试集中实际只有 1,000 张包。1,000 张衬衫中有 874 张被判为包。验证集上的同一选块策略准确率也只有 29.28%。这组结果表明，用完整输入训练的这组权重对高能量筛选很敏感。筛选可能改变了模型所依赖的明暗和轮廓线索；要确认具体原因，还需要单独的对照实验。

## 查看与复现

在本目录执行。仓库附带训练结果。以下第一条命令将 Fashion-MNIST 下载至 data/raw 并校验，已有完整数据时会直接复用。第二条使用已保存的最佳权重复测，更新 results 下的评测表和图。

```sh
../.venv/bin/python fashion_data.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 ../.venv/bin/python evaluate.py
```

从头训练并评测可使用新目录，保留本次结果。

```sh
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
../.venv/bin/python train.py --out results-rerun
../.venv/bin/python evaluate.py --out results-rerun
```

其他机器可在支持 MindSpore 2.6 的 Python 环境执行 `python -m pip install -r requirements.txt`，将上面的解释器路径改为该环境的 Python。若要用全部训练候选样本，给训练命令加上 `--per-class 5400`，仍会留出固定的 6,000 张验证图。两条脚本都支持 `--help`。

## 实验设计

- 数据集共 70,000 张 28×28 灰度图。按类别从官方训练集抽 600 张作验证，再从每类剩余样本抽 1,000 张作训练，保存原始编号。官方测试集 10,000 张仅在权重确定后评估。[数据集与 MD5 校验值](https://github.com/zalandoresearch/fashion-mnist#get-the-data)
- 将像素除以 255，切成 49 个互不重叠的 4×4 图块。每块投影至 64 维，加上原图索引对应的可学习位置向量。拼接可学习的 CLS 后，经过两个前置层归一化的 Transformer 编码层和分类头。每层有 4 个注意力头，前馈网络宽度为 128。
- 基准训练始终使用完整输入，批量 64，优化器为 AdamW，权重衰减 0.01。两个学习率 0.001 和 0.0003 各训 15 轮，共用划分和随机种子，以完整输入的验证准确率挑选一个检查点。
- 能量为块内 16 个像素的平方和。选择得分最高的 25 或 35 块，同分时优先保留原索引较小的块。随机对照每图无放回选 25 块，采用三个固定种子。中央对照取 7×7 网格内侧的 5×5 块。所有策略都只向编码器传递保留的图块和 CLS。
- 每种条件都对相同 300 张测试图逐张计时，在其批量 1 的输入形状上先预热 30 次。计时从内存中单张灰度图开始，包含归一化、切块、筛选、张量转换、推理和取最高分类分数对应的类别。每组报告中位数，随机组另报告三次结果的均值和样本标准差。

## 输出文件

| 文件 | 用途 |
|---|---|
| results/summary.csv、test_runs.csv | 各策略的准确率、宏平均 F1、单图延迟及随机种子明细 |
| results/training.png、history.csv | 学习率对照、训练损失与验证准确率 |
| results/best.ckpt、best.json、config.json、split.npz | 最佳权重、选择依据、训练配置和样本划分 |
| results/validation.csv | 选定权重在验证集上的完整输入及两档能量筛选结果 |
| results/per_class.csv、confusion.npz、confusion.png | 每类召回率和 F1、各条件的混淆矩阵，以及完整输入与能量 25 块的对照图 |
| results/example.png、predictions.npz | 同图不同策略的预测和每张测试图的真实标签、预测类别。示例优先展示完整输入正确且能量 25 块出错的衬衫 |
| results/latency.csv、benchmark.json | 每次计时的样本编号、毫秒数及运行环境 |

代码入口分别为 fashion_data.py（下载与选块）、model.py（模型）、train.py（训练）和 evaluate.py（评测与绘图）。
