# 不止于波形保真度

[English](README.md)

本仓库是论文 **Beyond Waveform Fidelity: A Low-Dimensional Spectral View of Denoised Vibration Signals for Fault Diagnosis** 的实验复现代码。

波形恢复得更好，不代表故障更容易识别。本文固定**同一批降噪输出**，比较完整包络谱特征（Full）、由干净训练特征构建的低维 POD 表示，以及同维随机投影（Random）。每种表示的分类器均单独用干净数据训练，测试时保持固定。实验改变的是用于诊断的表示，不是降噪后的波形。

## 实验范围

| 数据集 | 命令行 ID | 类别数 | 训练 / 验证 / 测试 | 用途 |
|---|---|---:|---:|---|
| 华中科技大学轴承（HUST） | `hust` | 9 | 531 / 108 / 108 | 正文实验 |
| 北京交通大学齿轮箱（BJTU-RAO） | `bjtu_rao` | 6 | 870 / 180 / 180 | 正文实验 |
| 哈工大航空发动机中介轴承（HIT） | `hit` | 4 | 540 / 162 / 162 | 正文实验 |
| 东北大学齿轮箱 | `northeastern_university` | 13 | 1,352 / 286 / 286 | 仅干净信号分析 |
| 渥太华大学轴承 | `ottawa` | 9 | 855 / 180 / 180 | 仅干净信号分析 |

正文实验使用 4,096 点信号窗口，比较 ResCNN、U-Net、U-WNO、Patch Mamba、Xformer-1D 和 ARDiff-RMS。每个模型训练 100 轮，使用五个随机种子（41–45），各次训练使用相同的数据划分与测试噪声。均值和标准差来自五次训练，不是五折交叉验证。测试输入信噪比为 0、−2、−4 和 −6 dB。数据出处和 Xformer-1D、ARDiff-RMS 的实现说明见[来源文档](DATA_AND_CODE_SOURCES.md)。

## 环境与目录

参考环境为 Python 3.9.25 和 PyTorch 2.7.1（CUDA 11.8）。先安装适合本机的 PyTorch，再进入代码目录执行：

```bash
python -m pip install -r requirements.txt
python reproduce.py verify --code-only --hash
```

Git 仓库只包含代码、训练配置和数据划分说明，**不包含信号数组、模型权重、预测缓存、评估器和结果表**。脚本默认在代码目录旁查找 `weights_datasets_results/`；实验文件放在其他位置或使用其他目录名时，设置环境变量 `PAPER_ASSET_ROOT`。

```text
code/                      # 上传 GitHub 的代码目录
weights_datasets_results/  # 独立实验文件，不提交到 Git
```

## 准备数据与训练

从[数据提供方](DATA_AND_CODE_SOURCES.md)获取原始记录。以下以 HIT 为例，先复建论文划分和固定白噪声测试输入，再做干净信号的低维可分性分析：

```bash
python prepare_data.py --dataset hit --raw-root "/path/to/HIT-recordings" --test-noise
python reproduce.py clean --dataset hit --all-ranks
```

[`protocols/`](protocols/) 保存切窗位置和标签。脚本会核对重建后的数据数组是否与论文实验一致。`--test-noise` 适用于三个正文数据集。干净信号分析不需要降噪权重；准备好数据后，也可以按 [`configs/`](configs/) 从头训练：

```bash
python reproduce.py train --dataset hit --model rescnn --seed 42 --epochs 100 --device cuda
```

新试验写到实验文件目录下的 `reruns/`，不覆盖原始权重。

## 复算已完成的实验

实验文件齐全时，可以先校验两个目录，再复算一组结果：

```bash
python reproduce.py verify --hash
python reproduce.py evaluate --dataset hit --model rescnn --seed 42 --snr -6
```

`evaluate` 默认从已保存的降噪信号复算指标，CPU 即可。加 `--infer --device cuda` 会加载权重重新推理。不同硬件上的重训与推理不保证逐位一致。

- `--representation`：`envelope`（包络谱）、`fft`（幅度谱）或 `time`（时域）。
- `--rank`：观察维数，可选 4、8、16、32、64、128，默认为 32。
- `--reduction sparse` 或 `--reduction kernel`：使用 HUST、BJTU-RAO 上预拟合的 32 维包络谱评估器。
- `--variant pink`、`--variant nonstationary` 或 `--variant output_constraint`：选择 HUST 的 ResCNN 扩展实验。非平稳噪声只评估 −6 dB。

## 文件索引

| 位置 | 内容 |
|---|---|
| [`models/`](models/)、[`reproduce.py`](reproduce.py) | 降噪模型与实验入口 |
| [`prepare_data.py`](prepare_data.py)、[`protocols/`](protocols/)、[`configs/`](configs/) | 数据准备脚本、固定划分和训练配置 |
| 实验目录的 `datasets/`、`weights/` | 数据数组和模型权重；信号为 `[N, 1, 4096]` 的 float32，标签为 `[N]` 的 int64 |
| 实验目录的 `predictions/`、`observers/` | 已保存的降噪输出和预拟合评估器；只加载可信的 `joblib` 文件 |
| 实验目录的 `results/six_model_per_run.csv` | 正文 360 条逐次结果，包含降噪和诊断指标 |
| 实验目录的 `results/six_model_summary_percent.csv`、`results/six_model_minus6_summary_percent.csv` | 不同信噪比及 −6 dB 的五次训练汇总 |
| 实验目录的 `results/supplement/`、`results/clean_supplement/` | 补充材料表格与干净信号分析 |

逐次结果表中的准确率是 0 到 1 的比例，正文汇总表用百分比。[`manifest.json`](manifest.json) 记录样本数量与相对于实验目录的权重路径。分享数据或权重前，请核对[再分发说明](DATA_AND_CODE_SOURCES.md)。
