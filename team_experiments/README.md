# Fashion-MNIST 四人实验交接说明

每张 28×28 的服饰图被切成 49 个 4×4 图块。四项实验分别研究删块造成的错误、保留哪些位置、怎样训练模型适应缺块，以及端到端能节省多少推理时间。

## 实验结论

| 实验 | 数据与结论 |
|---|---|
| 一 · 错误诊断 | 验证集上，选能量最高的 25 块使 3,353 张原本分类正确的图变成错判，其中 2,934 张被判成包。600 张真衬衫按亮度四等分后，从最暗到最亮，各组误判为包的张数为 82、140、146、150，每组 150 张。 |
| 二 · 选块算法 | 固定仓库权重，棋盘布局经验证集选出，在官方测试集答对 7,870/10,000 张。每个区域选能量最高的一块达到 58.18%，全局选能量最高的 25 块为 29.69%。|
| 三 · 训练策略 | 在同一份初始化和 2,355 次更新下，混合训练将棋盘输入的测试准确率从 78.70% 提到 82.58%（多答对 388 张），完整输入从 83.75% 变为 83.47%。两组均选中第 14 轮。本实验使用一个训练种子。|
| 四 · CPU 性能 | 在混合权重上，完整和棋盘输入的端到端 P50 分别为 0.575 和 0.509 ms，P95 分别为 0.769 和 0.686 ms。五轮中棋盘均更快，配对加速比中位数为 1.130×。|

如果优先保证准确率，本次比较中完整输入的最高测试准确率是 **83.75%**。若单张图只能保留 25 块，**混合训练 + 棋盘输入**取得 **82.58%** 和本机 **0.509 ms** 的中位延迟。两种选择相差 117/10,000 张测试图。延迟数字适用于所测 CPU 四线程、单图推理场景。

## 每位成员的代码与交付物

| 成员 | 代码入口 | 分析报告 | 可用于复核或答辩的材料 |
|---|---|---|---|
| 一 · 错误诊断 | [diagnosis.py](diagnosis.py) | [验证集错误与亮度分组](results/01_diagnosis/README.md) | [6,000 条逐图记录](results/01_diagnosis/validation_predictions.csv) · [同图掩码对照](results/01_diagnosis/examples.png) |
| 二 · 选块算法 | [selection.py](selection.py) · [选块实现](../fashion_data.py) | [不同布局的分类结果](results/02_selection/README.md) | [官方测试汇总](results/02_selection/test_summary.csv) · [选块图](results/02_selection/selection_masks.png) |
| 三 · 训练策略 | [training.py](training.py) · [训练步骤](../train.py) | [两种训练方式的对照](results/03_training/README.md) | [验证和测试的 2×2 对照](results/03_training/evaluation.csv) · [训练曲线](results/03_training/learning_curves.png) · [两组权重与训练记录](results/03_training/) |
| 四 · CPU 性能 | [performance.py](performance.py) · [单图计时函数](../evaluate.py) | [准确率、延迟与轮间波动](results/04_performance/README.md) | [12,000 条基线计时](results/04_performance/timings.csv) · [3,000 条混合权重计时](results/04_performance/mixed_pair/timings.csv) · [准确率–延迟图](results/04_performance/accuracy_latency.png) |

## 共同实验条件

实验在 Windows x64、AMD Ryzen 7 9800X3D 上运行，使用 Python 3.10.5、MindSpore 2.6.0、NumPy 1.26.4 和 Matplotlib 3.9.4。默认使用 4 个 CPU 线程。Fashion-MNIST 原始数据由项目下载并逐文件校验 MD5。

成员一、二、四统一加载仓库的 [best.ckpt](../results/best.ckpt)。成员三从同一份随机初始化出发训练两个模型，使用相同的样本顺序、优化步数和检查点选择标准。训练用 10,000 张，验证用 6,000 张，每类分别为 1,000、600 张，均由 [split.npz](../results/split.npz) 固定。官方测试集另有 10,000 张。错误诊断用验证集，选块方案和训练轮次也在验证集确定。

## 在本机重跑

在仓库根目录打开 PowerShell。以下命令分别生成一份新结果，末条命令用刚训练的混合权重补测速度。性能测试请等其余训练、评估全部退出后单独运行。

```powershell
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MPLCONFIGDIR = Join-Path (Get-Location) '.venv\mpl-cache'
.\.venv\Scripts\python.exe -m team_experiments.diagnosis --out results-rerun-team/01_diagnosis
.\.venv\Scripts\python.exe -m team_experiments.selection --out results-rerun-team/02_selection
.\.venv\Scripts\python.exe -m team_experiments.training --out results-rerun-team/03_training
.\.venv\Scripts\python.exe -m team_experiments.performance --out results-rerun-team/04_performance
.\.venv\Scripts\python.exe -m team_experiments.performance --mixed-pair --checkpoint results-rerun-team/03_training/best_mixed.ckpt --out results-rerun-team/04_performance/mixed_pair
```

在新机器解压时，先进入含 `requirements.txt` 的目录，创建虚拟环境并安装依赖。首次运行会拉取四个数据文件；也可以把已有的 `data/raw` 原样复制到项目中。Windows x64、CPython 3.10 若遇到 pip 找不到指定 MindSpore 版本，可以使用本机验证过的官方 PyPI 文件地址。

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install 'https://files.pythonhosted.org/packages/16/23/cf470c5553d585d4fb295e7dec2e331317b80d69195600e2085514fc9588/mindspore-2.6.0-cp310-cp310-win_amd64.whl#sha256=89d2cab829c9214b257e7df15eda8fed7e1281aa522535f3c6ff427e295c2d43' -r requirements.txt
```
