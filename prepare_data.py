"""Rebuild the exact paper windows and test noise using the bundled protocols."""

import argparse
import hashlib
import io

import numpy as np
import pandas as pd

from reproduce import ASSET_ROOT, DATASETS, ROOT, SNRS, SPLITS, dataset_arrays, load_json


def raw_loader(dataset, source):
    if dataset == "hust":
        return pd.read_csv(source, header=None, skiprows=22, usecols=[2]).to_numpy().ravel().astype(np.float32)
    if dataset == "bjtu_rao":
        return pd.read_csv(source, usecols=["CH11"]).to_numpy().ravel().astype(np.float32)
    if dataset == "hit":
        return np.load(source, mmap_mode="r", allow_pickle=False)
    if dataset == "northeastern_university":
        return np.loadtxt(source, delimiter=",", skiprows=4, usecols=0, dtype=np.float32)
    return pd.read_csv(source, usecols=["Accelerometer"])["Accelerometer"].to_numpy(dtype=np.float32)


def save_exact(path, array, expected=None):
    encoded = io.BytesIO()
    np.save(encoded, array, allow_pickle=False)
    content = encoded.getbuffer()
    if expected is not None and hashlib.sha256(content).hexdigest() != expected:
        raise ValueError(f"Regenerated array does not match the paper protocol: {path}")
    if path.exists():
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), array)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def prepare(dataset, raw_root):
    root = ASSET_ROOT / "datasets" / dataset
    protocol_dir = ROOT / "protocols" / dataset
    protocol = load_json(protocol_dir / "data_protocol.json")
    metadata = pd.read_csv(protocol_dir / "sample_metadata.csv")
    files = {}
    for name in metadata["source_file"].unique():
        matches = list(raw_root.rglob(name))
        if len(matches) != 1:
            raise ValueError(f"Expected one copy of {name} under the supplied raw root, found {len(matches)}.")
        files[name] = raw_loader(dataset, matches[0])
    for split in SPLITS:
        rows = metadata[metadata["split"] == split]
        windows = []
        for row in rows.itertuples():
            signal = files[row.source_file]
            if dataset == "hit":
                signal = signal[int(row.source_row), int(row.channel_1based) - 1]
            windows.append(np.asarray(signal[row.start:row.end], dtype=np.float32))
        arrays = {"x": np.stack(windows)[:, None, :], "y": rows["label"].to_numpy(dtype=np.int64)}
        for name, array in arrays.items():
            path = f"{split}/{name}.npy"
            save_exact(root / path, array, protocol["array_sha256"][path])
    print(f"Verified paper windows: {dataset}")


def test_noise(dataset):
    import torch
    from run_pod_denoising_experiments import make_noisy

    clean, _ = dataset_arrays(dataset, "test")
    offset = 2_000_000 if dataset == "hit" else 0
    split = ASSET_ROOT / "datasets" / dataset / "test_inputs/white"
    for snr in SNRS:
        noisy = np.concatenate([
            make_noisy(torch.from_numpy(clean[start:start + 64]), snr, offset + block * 1000).numpy()
            for block, start in enumerate(range(0, len(clean), 64))])
        save_exact(split / f"snr_{snr}.npy", noisy)
    print(f"Verified white test inputs: {dataset}")


if __name__ == "__main__":
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--raw-root", type=Path)
    parser.add_argument("--test-noise", action="store_true")
    args = parser.parse_args()
    if args.raw_root is None and not args.test_noise:
        parser.error("Provide --raw-root, --test-noise, or both.")
    if args.raw_root is not None:
        prepare(args.dataset, args.raw_root)
    if args.test_noise:
        if args.dataset not in DATASETS[:3]:
            parser.error("White test-noise references exist only for the three denoising datasets.")
        test_noise(args.dataset)
