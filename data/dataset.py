from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .noise import add_awgn


def _load_npy(path, key=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing npy file: {path.resolve()}")
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.lib.npyio.NpzFile):
        if key is None:
            keys = list(obj.keys())
            if len(keys) != 1:
                raise ValueError(f"{path} contains keys {keys}; set the config key explicitly.")
            key = keys[0]
        return obj[key]
    if obj.shape == () and isinstance(obj.item(), dict):
        data = obj.item()
        if key is None:
            candidates = [k for k in ("signals", "data", "x", "X") if k in data]
            if not candidates:
                raise ValueError(f"{path} stores a dict; set data_key/label_key in config.")
            key = candidates[0]
        return data[key]
    return obj


def _load_data_and_optional_labels(path, data_key=None, label_key=None):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing npy file: {path.resolve()}")
    obj = np.load(path, allow_pickle=True)
    if isinstance(obj, np.lib.npyio.NpzFile):
        keys = list(obj.keys())
        if data_key is None:
            candidates = [k for k in ("signals", "data", "x", "X") if k in obj]
            data_key = candidates[0] if candidates else (keys[0] if len(keys) == 1 else None)
        if data_key is None:
            raise ValueError(f"{path} contains keys {keys}; set data_key in config.")
        labels = obj[label_key] if label_key and label_key in obj else None
        if labels is None:
            for key in ("labels", "label", "y", "Y"):
                if key in obj:
                    labels = obj[key]
                    break
        return obj[data_key], labels
    if obj.shape == () and isinstance(obj.item(), dict):
        data = obj.item()
        if data_key is None:
            candidates = [k for k in ("signals", "data", "x", "X") if k in data]
            if not candidates:
                raise ValueError(f"{path} stores a dict; set data_key in config.")
            data_key = candidates[0]
        labels = data.get(label_key) if label_key else None
        if labels is None:
            for key in ("labels", "label", "y", "Y"):
                if key in data:
                    labels = data[key]
                    break
        return data[data_key], labels
    return obj, None


def canonicalize_signals(array, pair_mode="first"):
    """Convert [N,L], [N,1,L], or [N,2,L] into [N,1,L]."""
    x = np.asarray(array, dtype=np.float32)
    if x.ndim == 2:
        x = x[:, None, :]
    elif x.ndim == 3:
        if x.shape[1] == 1:
            pass
        elif x.shape[1] == 2:
            if pair_mode not in {"first", "second", "clean", "noisy"}:
                raise ValueError("signal_pair_mode must be first/second/clean/noisy for [N,2,L] data.")
            channel = 0 if pair_mode in {"first", "clean"} else 1
            x = x[:, channel : channel + 1, :]
        else:
            raise ValueError(f"Expected channel dimension 1 or 2 for [N,C,L], got shape {x.shape}.")
    else:
        raise ValueError(f"Expected signal shape [N,L], [N,1,L], or [N,2,L], got {x.shape}.")
    if x.shape[0] == 0 or x.shape[-1] < 2:
        raise ValueError(f"Invalid signal array shape after canonicalization: {x.shape}.")
    return x


def _load_numeric_csv_column(path, column=1, skip_header=0):
    arr = np.genfromtxt(path, delimiter=",", dtype=np.float32, invalid_raise=False, skip_header=int(skip_header))
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.size == 0:
        raise ValueError(f"No numeric rows found in CSV file: {path}")
    valid = np.isfinite(arr).any(axis=1)
    arr = arr[valid]
    if arr.shape[1] <= int(column):
        raise ValueError(f"CSV {path} has {arr.shape[1]} numeric columns, cannot read column {column}.")
    sig = arr[:, int(column)]
    sig = sig[np.isfinite(sig)]
    if len(sig) < 2:
        raise ValueError(f"CSV {path} column {column} does not contain a valid signal.")
    return sig.astype(np.float32)


def _label_name_from_csv(path, strip_speed=True):
    stem = Path(path).stem
    if strip_speed:
        parts = stem.split("_")
        if parts and parts[-1].lower().endswith("hz"):
            return "_".join(parts[:-1])
    return stem


def _load_label_map_from_counts(csv_dir):
    counts_path = Path(csv_dir).parent / "class_counts.csv"
    if not counts_path.exists():
        return None
    try:
        table = np.genfromtxt(counts_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
    except Exception:
        return None
    names = {name.lstrip("\ufeff"): name for name in table.dtype.names}
    if table.size == 0 or "label_name" not in names or "class_label" not in names:
        return None
    table = np.atleast_1d(table)
    return {str(row[names["label_name"]]): int(row[names["class_label"]]) for row in table}


def _make_segments(signal, segment_length, segment_stride):
    length = int(segment_length)
    stride = int(segment_stride or segment_length)
    if len(signal) < length:
        raise ValueError(f"Signal length {len(signal)} is shorter than segment_length {length}.")
    starts = np.arange(0, len(signal) - length + 1, stride, dtype=np.int64)
    return np.stack([signal[s : s + length] for s in starts]).astype(np.float32)


def split_indices(labels, split, ratios, split_by_label=True, time_ordered=True, split_purge=0):
    if split not in {"train", "val", "test"}:
        raise ValueError(f"split must be train/val/test, got {split}.")
    n = len(labels)
    base = np.arange(n)
    groups = []
    if labels is not None and split_by_label:
        for label in np.unique(labels):
            groups.append(base[labels == label])
    else:
        groups.append(base)

    out = []
    train_r = float(ratios["train"])
    val_r = float(ratios["val"])
    purge = int(split_purge or 0) if time_ordered else 0
    for idx in groups:
        idx = np.asarray(idx)
        if not time_ordered:
            rng = np.random.default_rng(0)
            idx = rng.permutation(idx)
        n_group = len(idx)
        n_train = int(np.floor(n_group * train_r))
        n_val = int(np.floor(n_group * val_r))
        if split == "train":
            part = idx[:n_train]
        elif split == "val":
            part = idx[n_train + purge : n_train + n_val]
        else:
            part = idx[n_train + n_val + purge :]
        out.append(part)
    result = np.concatenate(out) if out else np.array([], dtype=np.int64)
    return np.sort(result) if time_ordered else result


class VibrationDataset(Dataset):
    def __init__(
        self,
        data_path,
        labels_path=None,
        split="train",
        split_ratios=None,
        split_by_label=True,
        time_ordered=True,
        split_purge=0,
        data_key=None,
        label_key=None,
        signal_pair_mode="first",
        snr_db=None,
        seed=42,
        return_labels=True,
    ):
        split_ratios = split_ratios or {"train": 0.7, "val": 0.15, "test": 0.15}
        raw, embedded_labels = _load_data_and_optional_labels(data_path, data_key, label_key)
        signals = canonicalize_signals(raw, pair_mode=signal_pair_mode)
        labels = None
        if labels_path:
            labels = np.asarray(_load_npy(labels_path, label_key)).reshape(-1)
            if len(labels) != len(signals):
                raise ValueError(f"labels length {len(labels)} does not match signals length {len(signals)}.")
        elif embedded_labels is not None:
            labels = np.asarray(embedded_labels).reshape(-1)
            if len(labels) != len(signals):
                raise ValueError(f"embedded labels length {len(labels)} does not match signals length {len(signals)}.")
        indices = split_indices(labels, split, split_ratios, split_by_label, time_ordered, split_purge)
        if len(indices) == 0:
            raise ValueError(f"No samples in {split} split. Check split ratios and class counts.")
        self.signals = signals[indices]
        self.labels = labels[indices] if labels is not None else None
        self.indices = indices
        self.snr_db = snr_db
        self.return_labels = return_labels
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = self.signals[idx].copy()
        if self.snr_db is not None:
            x = add_awgn(x, self.snr_db, self.rng)
        item = {"signal": torch.from_numpy(x), "index": int(self.indices[idx])}
        if self.return_labels and self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item


class CsvFolderVibrationDataset(Dataset):
    def __init__(
        self,
        data_path,
        split="train",
        split_ratios=None,
        time_ordered=True,
        split_purge=0,
        segment_length=2048,
        segment_stride=None,
        csv_signal_column=1,
        csv_skip_rows=0,
        csv_label_strip_speed=True,
        snr_db=None,
        seed=42,
        return_labels=True,
    ):
        split_ratios = split_ratios or {"train": 0.7, "val": 0.15, "test": 0.15}
        data_dir = Path(data_path)
        if not data_dir.exists() or not data_dir.is_dir():
            raise FileNotFoundError(f"CSV data_path must be a folder: {data_dir.resolve()}")
        csv_files = sorted(data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in folder: {data_dir.resolve()}")

        label_map = _load_label_map_from_counts(data_dir)
        if label_map is None:
            names = [_label_name_from_csv(p, csv_label_strip_speed) for p in csv_files]
            label_map = {name: i for i, name in enumerate(sorted(set(names)))}

        signals, labels, indices, names_out = [], [], [], []
        global_index = 0
        for csv_path in csv_files:
            name = _label_name_from_csv(csv_path, csv_label_strip_speed)
            label = label_map.get(name)
            if label is None:
                label = len(label_map)
                label_map[name] = label
            raw = _load_numeric_csv_column(csv_path, csv_signal_column, csv_skip_rows)
            segments = _make_segments(raw, segment_length, segment_stride)
            local_labels = np.full(len(segments), int(label), dtype=np.int64)
            local_idx = split_indices(
                local_labels,
                split,
                split_ratios,
                split_by_label=False,
                time_ordered=time_ordered,
                split_purge=split_purge,
            )
            if len(local_idx) == 0:
                continue
            selected = segments[local_idx]
            signals.append(selected[:, None, :])
            labels.append(local_labels[local_idx])
            indices.append(np.arange(global_index, global_index + len(selected), dtype=np.int64))
            names_out.extend([name] * len(selected))
            global_index += len(selected)

        if not signals:
            raise ValueError(f"No samples in {split} split. Check CSV segment and split settings.")
        self.signals = np.concatenate(signals, axis=0).astype(np.float32)
        self.labels = np.concatenate(labels, axis=0).astype(np.int64)
        self.indices = np.concatenate(indices, axis=0)
        self.label_names = names_out
        self.label_map = label_map
        self.snr_db = snr_db
        self.return_labels = return_labels
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = self.signals[idx].copy()
        if self.snr_db is not None:
            x = add_awgn(x, self.snr_db, self.rng)
        item = {"signal": torch.from_numpy(x), "index": int(self.indices[idx])}
        if self.return_labels:
            item["label"] = int(self.labels[idx])
        return item


class SplitNpyVibrationDataset(Dataset):
    def __init__(
        self,
        data_path,
        split="train",
        split_file_map=None,
        snr_db=None,
        seed=42,
        return_labels=True,
        signal_pair_mode="first",
    ):
        data_dir = Path(data_path)
        split_name = (split_file_map or {}).get(split, split)
        split_dir = data_dir / split_name
        x_path = split_dir / "x.npy"
        y_path = split_dir / "y.npy"
        if not x_path.exists():
            raise FileNotFoundError(f"Missing split x.npy: {x_path.resolve()}")
        raw = _load_npy(x_path)
        signals = canonicalize_signals(raw, pair_mode=signal_pair_mode)
        labels = None
        if y_path.exists():
            labels = np.asarray(_load_npy(y_path)).reshape(-1)
            if len(labels) != len(signals):
                raise ValueError(f"labels length {len(labels)} does not match signals length {len(signals)}.")
        if len(signals) == 0:
            raise ValueError(f"No samples in split_npy split: {split_name}")
        self.signals = signals
        self.labels = labels
        self.indices = np.arange(len(signals), dtype=np.int64)
        self.snr_db = snr_db
        self.return_labels = return_labels
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        x = self.signals[idx].copy()
        if self.snr_db is not None:
            x = add_awgn(x, self.snr_db, self.rng)
        item = {"signal": torch.from_numpy(x), "index": int(self.indices[idx])}
        if self.return_labels and self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item


def build_dataloader(config, split="train", shuffle=None):
    data_cfg = dict(config["data"])
    shuffle = (split == "train" and not data_cfg.get("time_ordered", True)) if shuffle is None else shuffle
    source_type = data_cfg.get("source_type", "npy")
    if source_type == "split_npy":
        dataset = SplitNpyVibrationDataset(
            data_path=data_cfg["data_path"],
            split=split,
            split_file_map=data_cfg.get("split_file_map"),
            snr_db=data_cfg.get("snr_db"),
            seed=config.get("seed", 42),
            return_labels=data_cfg.get("return_labels", True),
            signal_pair_mode=data_cfg.get("signal_pair_mode", "first"),
        )
    elif source_type == "csv_folder":
        dataset = CsvFolderVibrationDataset(
            data_path=data_cfg["data_path"],
            split=split,
            split_ratios=data_cfg.get("split_ratios"),
            time_ordered=data_cfg.get("time_ordered", True),
            split_purge=data_cfg.get("split_purge", 0),
            segment_length=data_cfg.get("segment_length", 2048),
            segment_stride=data_cfg.get("segment_stride"),
            csv_signal_column=data_cfg.get("csv_signal_column", 1),
            csv_skip_rows=data_cfg.get("csv_skip_rows", 0),
            csv_label_strip_speed=data_cfg.get("csv_label_strip_speed", True),
            snr_db=data_cfg.get("snr_db"),
            seed=config.get("seed", 42),
            return_labels=data_cfg.get("return_labels", True),
        )
    else:
        dataset = VibrationDataset(
            data_path=data_cfg["data_path"],
            labels_path=data_cfg.get("labels_path"),
            split=split,
            split_ratios=data_cfg.get("split_ratios"),
            split_by_label=data_cfg.get("split_by_label", True),
            time_ordered=data_cfg.get("time_ordered", True),
            split_purge=data_cfg.get("split_purge", 0),
            data_key=data_cfg.get("data_key"),
            label_key=data_cfg.get("label_key"),
            signal_pair_mode=data_cfg.get("signal_pair_mode", "first"),
            snr_db=data_cfg.get("snr_db"),
            seed=config.get("seed", 42),
            return_labels=data_cfg.get("return_labels", True),
        )
    return DataLoader(
        dataset,
        batch_size=int(data_cfg.get("batch_size", 64)),
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        drop_last=False,
    )
