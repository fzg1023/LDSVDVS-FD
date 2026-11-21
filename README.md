# Beyond Waveform Fidelity

[简体中文](README.zh-CN.md)

Research code for **Beyond Waveform Fidelity: A Low-Dimensional Spectral View of Denoised Vibration Signals for Fault Diagnosis**.

A denoiser that reconstructs the reference waveform more closely does not necessarily produce signals that are easier to classify. We evaluate the **same denoised outputs** using full envelope-spectrum features (Full), a low-dimensional representation learned from clean training features (POD), and dimension-matched random projections (Random). A separate classifier is trained on clean data for each representation and held fixed during testing. The comparison changes the representation used for diagnosis, not the waveform.

## Experiments

| Dataset | Command-line ID | Classes | Train / Validation / Test | Scope |
|---|---|---:|---:|---|
| HUST bearing | `hust` | 9 | 531 / 108 / 108 | Main |
| BJTU-RAO gearbox | `bjtu_rao` | 6 | 870 / 180 / 180 | Main |
| HIT aero-engine inter-shaft bearing | `hit` | 4 | 540 / 162 / 162 | Main |
| Northeastern University gearbox | `northeastern_university` | 13 | 1,352 / 286 / 286 | Clean signals only |
| University of Ottawa bearing | `ottawa` | 9 | 855 / 180 / 180 | Clean signals only |

The main experiments use 4,096-point windows and six denoisers: ResCNN, U-Net, U-WNO, Patch Mamba, Xformer-1D, and ARDiff-RMS. Each model is trained for 100 epochs with five seeds (41–45). The splits and test noise are fixed across runs. Reported means and standard deviations describe repeated training runs, not cross-validation folds. Input test SNRs are 0, −2, −4, and −6 dB. See the [source notes](DATA_AND_CODE_SOURCES.md) for dataset attribution and the Xformer-1D and ARDiff-RMS adaptations.

## Setup

The reference environment is Python 3.9.25 and PyTorch 2.7.1 with CUDA 11.8. Install a PyTorch build appropriate for your hardware, then run these commands from the repository:

```bash
python -m pip install -r requirements.txt
python reproduce.py verify --code-only --hash
```

The Git repository contains the code, training configs, and split protocols. **It does not contain the signal arrays, checkpoints, prediction caches, fitted observers, or result tables.** The scripts look for `weights_datasets_results/` next to the repository by default. Set `PAPER_ASSET_ROOT` to use another location or folder name.

```text
code/                      # Git repository
weights_datasets_results/  # Experiment files, outside Git
```

## Prepare data and train

Obtain the source recordings from the [data providers](DATA_AND_CODE_SOURCES.md). The following example rebuilds the HIT splits and the fixed white-noise test inputs, then runs the clean-signal observation analysis:

```bash
python prepare_data.py --dataset hit --raw-root "/path/to/HIT-recordings" --test-noise
python reproduce.py clean --dataset hit --all-ranks
```

Window indices and labels are recorded in [`protocols/`](protocols/). The preparation script checks the regenerated split arrays against their reference hashes. The `--test-noise` option is available for the three main datasets. Clean-signal analysis needs no pretrained denoiser. Once the data splits are prepared, the configs in [`configs/`](configs/) can also be used to train from scratch:

```bash
python reproduce.py train --dataset hit --model rescnn --seed 42 --epochs 100 --device cuda
```

New runs are written to `reruns/` under the asset root. The original checkpoints are not overwritten.

## Evaluate saved experiments

With the complete asset folder available, verify both bundles and evaluate one run:

```bash
python reproduce.py verify --hash
python reproduce.py evaluate --dataset hit --model rescnn --seed 42 --snr -6
```

The command reads saved denoised signals by default. Add `--infer --device cuda` to run the checkpoint instead. CPU is sufficient for evaluation from saved signals. Results from fresh training or inference are not expected to be bit-identical across hardware.

- `--representation` selects `envelope`, `fft` (magnitude spectrum), or `time`.
- `--rank` selects the observation dimension: 4, 8, 16, 32, 64, or 128. The default is 32.
- `--reduction sparse` and `--reduction kernel` use the fitted 32-dimensional envelope-spectrum observers for HUST and BJTU-RAO.
- `--variant pink`, `--variant nonstationary`, and `--variant output_constraint` select the HUST ResCNN extensions. Nonstationary noise is evaluated at −6 dB only.

## Find the files

| Location | Contents |
|---|---|
| [`models/`](models/), [`reproduce.py`](reproduce.py) | Denoisers and experiment entry point |
| [`prepare_data.py`](prepare_data.py), [`protocols/`](protocols/), [`configs/`](configs/) | Data preparation, fixed splits, and training configurations |
| Asset folder: `datasets/`, `weights/` | Split arrays and model checkpoints; signals are float32 `[N, 1, 4096]` and labels are int64 `[N]` |
| Asset folder: `predictions/`, `observers/` | Saved denoised signals and fitted observers; load only trusted `joblib` files |
| Asset folder: `results/six_model_per_run.csv` | 360 main-experiment records with restoration and diagnosis metrics |
| Asset folder: `results/six_model_summary_percent.csv`, `results/six_model_minus6_summary_percent.csv` | Five-run summaries across test SNRs and at −6 dB, respectively |
| Asset folder: `results/supplement/`, `results/clean_supplement/` | Supplementary tables and clean-signal analyses |

Accuracy in the per-run CSV is a fraction; the main summary CSVs report percentages. [`manifest.json`](manifest.json) records split counts and asset-relative checkpoint paths. Do not commit the asset folder to Git. Check the [redistribution notes](DATA_AND_CODE_SOURCES.md) before sharing data or weights.
