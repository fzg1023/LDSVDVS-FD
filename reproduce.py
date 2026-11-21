"""Reproduce the current manuscript's diagnostic experiments."""

import argparse
from contextlib import nullcontext
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

import observe_core as observation


ROOT = Path(__file__).resolve().parent
ASSET_ROOT = Path(os.environ.get("PAPER_ASSET_ROOT") or ROOT.with_name("weights_datasets_results")).expanduser().resolve()
DATASETS = ("hust", "bjtu_rao", "hit", "northeastern_university", "ottawa")
MODELS = ("rescnn", "unet", "uwno", "patch_mamba", "xformer_1d", "ardiff_rms")
SNRS = (0, -2, -4, -6)
RANKS = tuple(observation.RANKS)
SPLITS = ("train", "val", "test")
VARIANTS = ("main", "pink", "nonstationary", "output_constraint")
FREQUENCIES = dict(zip(DATASETS, (25600, 64000, 25000, 15360, 42000)))


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_arrays(dataset, split):
    directory = ASSET_ROOT / "datasets" / dataset / split
    return tuple(np.load(directory / f"{name}.npy", allow_pickle=False) for name in ("x", "y"))


def bank_for(dataset, representation):
    folder = ASSET_ROOT / "observers" / dataset / representation
    if (folder / "classifiers.joblib").is_file() and (folder / "random_bases.npy").is_file():
        return joblib.load(folder / "classifiers.joblib"), np.load(folder / "random_bases.npy", allow_pickle=False)
    train_x, train_y = dataset_arrays(dataset, "train")
    features = observation.features(train_x, representation, FREQUENCIES[dataset])
    pool = observation.random_pool(features.shape[1])
    return observation.fit_bank(features, train_y, pool, {"dataset": dataset}), pool


def role_of(args):
    if args.variant == "main":
        return args.model
    if args.dataset != "hust" or args.model != "rescnn":
        raise ValueError("The supplemental variants apply only to HUST/ResCNN.")
    return f"rescnn_{args.variant}"


def diagnose(bank, pool, values, labels, rank):
    centered = values.astype(np.float64) - bank["mean"]
    full = bank["full"].predict(values)
    pod = bank["pod"][rank].predict(centered @ bank["basis"][:rank].T)
    random = [classifier.predict(centered @ basis[:, :rank])
              for classifier, basis in zip(bank["random"][rank], pool)]
    return {"full_accuracy": float(np.mean(full == labels)), "pod_accuracy": float(np.mean(pod == labels)),
            "random_accuracy": float(np.mean(np.asarray(random) == labels))}


def alternative(args, values, labels):
    if args.dataset not in ("hust", "bjtu_rao") or args.representation != "envelope" or args.rank != 32:
        raise ValueError("The matched alternative observers are HUST/BJTU-RAO, envelope, rank 32.")
    method = {"sparse": "SparsePCA", "kernel": "KernelPCA_RBF"}[args.reduction]
    path = ASSET_ROOT / "observers" / args.dataset / method / "selected_bank.joblib"
    model = joblib.load(path)
    features = values.astype(np.float64) - model["preprocessor"]["input_mean"]
    coords = model["estimator"].transform(features)
    return float(np.mean(model["classifier"].predict(coords) == labels))


def scatter(values, labels, basis):
    overall = values.mean(axis=0)
    mean_differences, within = [], []
    for label in np.unique(labels):
        subset = values[labels == label]
        class_mean = subset.mean(axis=0)
        mean_differences.append((class_mean - overall) * np.sqrt(len(subset)))
        within.append(subset - class_mean)
    between = np.stack(mean_differences)
    within = np.concatenate(within, axis=0)
    between_ratio = float(np.sum((between @ basis) ** 2) / np.sum(between ** 2))
    within_ratio = float(np.sum((within @ basis) ** 2) / np.sum(within ** 2))
    return {"between_class_retention": between_ratio, "within_class_retention": within_ratio,
            "relative_gain": between_ratio / within_ratio}


def clean_result(args):
    raw, labels = dataset_arrays(args.dataset, "test")
    values = observation.features(raw, args.representation, FREQUENCIES[args.dataset])
    bank, pool = bank_for(args.dataset, args.representation)
    rows = []
    for rank in RANKS if args.all_ranks else (args.rank,):
        scatter_values = values.astype(np.float64)
        random_scatter = [scatter(scatter_values, labels, basis[:, :rank]) for basis in pool]
        rows.append({"dataset": args.dataset, "representation": args.representation, "rank": rank,
                     "explained_variance": float(bank["energy"][rank - 1]),
                     **diagnose(bank, pool, values, labels, rank),
                     **scatter(scatter_values, labels, bank["basis"][:rank].T),
                     **{f"random_{k}": float(np.mean([row[k] for row in random_scatter]))
                        for k in random_scatter[0]}})
    return rows


def build_model(name, cfg, device):
    from models.additional_baselines import PatchMamba1D
    from models.denoiser_rescnn_tcn import ResCNN1DDenoiser
    from models.pod_denoisers import UWNO1DDenoiser
    from models.unet_baseline import UNet1DBaseline
    from models.xformer_1d import Xformer1D

    base = cfg["model"].get("base_channels", 32)
    if name == "rescnn":
        model = ResCNN1DDenoiser(width=64, depth=8, kernel_size=7, residual=False)
    elif name == "unet":
        model = UNet1DBaseline(base_channels=base)
    elif name == "uwno":
        model = UWNO1DDenoiser(base_channels=base, width=128, depth=3, levels=3, residual=False)
    elif name == "patch_mamba":
        model = PatchMamba1D()
    elif name == "xformer_1d":
        keys = ("dim", "channel_blocks", "temporal_blocks", "refinement_blocks", "heads", "window", "expansion", "drop_path")
        model = Xformer1D(**{key: cfg["model"][key] for key in keys})
    else:
        raise ValueError(name)
    return model.to(device)


def device_for(request):
    import torch
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if request == "auto" else request)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    return device


def infer(args, folder, noisy):
    import torch
    device = device_for(args.device)
    cfg = load_json(folder / "config.json")
    checkpoint = torch.load(folder / "best.pth", map_location="cpu", weights_only=True)
    batch = args.batch_size
    if args.model == "ardiff_rms":
        from models.ardiff_equations import cosine_alpha_bar
        from models.ardiff_noisy_rms import denoise_noisy_rms
        from run_ardiff_noisy_rms import make_models
        settings = argparse.Namespace(**cfg)
        model, predictor = make_models(settings, device)
        model.load_state_dict(checkpoint["diffusion"])
        predictor.load_state_dict(checkpoint["step_predictor"])
        model.eval()
        predictor.eval()
        alpha = cosine_alpha_bar(settings.steps, settings.cosine_offset).to(device)
        batch = batch or settings.eval_batch_size

        def denoise(tensor):
            return denoise_noisy_rms(model, predictor, tensor, alpha,
                                     amp=settings.amp and device.type == "cuda",
                                     step_prediction_scale=settings.steps)[0]
    else:
        model = build_model(args.model, cfg, device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        batch = batch or 64

        def denoise(tensor):
            scale = tensor.square().mean(-1, keepdim=True).sqrt().clamp_min(1e-8)
            waveform, _ = model(tensor / scale, return_aux=True)
            return waveform * scale

    with torch.inference_mode():
        return np.concatenate([denoise(torch.from_numpy(noisy[i:i + batch]).to(device)).cpu().numpy()
                               for i in range(0, len(noisy), batch)])


def evaluate(args):
    from utils.metrics import frequency_metrics, waveform_metrics
    role = role_of(args)
    noise = args.variant if args.variant in ("pink", "nonstationary") else "white"
    clean, labels = dataset_arrays(args.dataset, "test")
    noisy = np.load(ASSET_ROOT / "datasets" / args.dataset / "test_inputs" / noise / f"snr_{args.snr}.npy", allow_pickle=False)
    suffix = Path(args.dataset) / role / f"seed_{args.seed}"
    if args.infer:
        denoised = infer(args, ASSET_ROOT / "weights" / suffix, noisy)
    else:
        denoised = np.load(ASSET_ROOT / "predictions" / suffix / f"snr_{args.snr}.npy", allow_pickle=False)
    if denoised.shape != clean.shape or noisy.shape != clean.shape or not np.isfinite(denoised).all():
        raise ValueError("The prediction, noisy and clean arrays must have matching finite signals.")
    fs = FREQUENCIES[args.dataset]
    values = observation.features(denoised, args.representation, fs)
    bank, pool = bank_for(args.dataset, args.representation)
    row = {"dataset": args.dataset, "model": args.model, "seed": args.seed, "snr_db": args.snr,
           "variant": args.variant, "representation": args.representation, "rank": args.rank,
           **waveform_metrics(clean, denoised, noisy)}
    for rep, key in (("envelope", "envelope_corr"), ("fft", "fft_corr")):
        row[key] = frequency_metrics(observation.features(clean, rep, fs), observation.features(denoised, rep, fs))["fpcc"]
    row.update(diagnose(bank, pool, values, labels, args.rank))
    if args.reduction != "pod":
        row[args.reduction + "_accuracy"] = alternative(args, values, labels)
    return row


def train(args):
    import torch
    from utils.seed import set_seed
    role = role_of(args)
    device = device_for(args.device)
    cfg = load_json(ROOT / "configs" / args.dataset / role / f"seed_{args.seed}" / "config.json")
    out = ASSET_ROOT / "reruns" / args.dataset / role / f"seed_{args.seed}_{datetime.now():%Y%m%dT%H%M%S%f}"
    out.mkdir(parents=True)
    set_seed(args.seed)
    if args.model == "ardiff_rms":
        import run_ardiff_reproduction as engine
        from models.ardiff_equations import cosine_alpha_bar
        from run_ardiff_noisy_rms import make_models, rms_training_callback
        cfg.update(epochs=args.epochs, amp=cfg["amp"] and device.type == "cuda")
        settings = argparse.Namespace(**cfg)
        model, predictor = make_models(settings, device)
        alpha = cosine_alpha_bar(settings.steps, settings.cosine_offset).to(device)
        arrays = {split: torch.from_numpy(dataset_arrays(args.dataset, split)[0]) for split in ("train", "val")}
        with rms_training_callback():
            engine.run_training(model, predictor, arrays, None, alpha, settings, device, out, cfg)
    else:
        import run_hust_pod_idea_tests as engine
        from run_pod_denoising_experiments import PODProjector
        cfg["data"]["data_path"] = str(ASSET_ROOT / "datasets" / args.dataset)
        cfg["device"] = str(device)
        cfg["training"]["epochs"] = args.epochs
        cfg["paths"]["pod_analysis_dir"] = str(ASSET_ROOT / "observers" / args.dataset)
        cfg["paths"]["pod_denoising_ablation_dir"] = str(out)
        engine.build_backbone = build_model
        if args.dataset == "hit":
            original_noise, original_eval = engine.make_noisy_for_model, engine.evaluate
            phase = {"offset": 0}

            def noise(clean, snrs, config, seed_offset=0):
                return original_noise(clean, snrs, config, seed_offset + phase["offset"])

            def validate(*values, **keywords):
                old = phase["offset"]
                phase["offset"] = 1_000_000
                try:
                    return original_eval(*values, **keywords)
                finally:
                    phase["offset"] = old

            engine.make_noisy_for_model, engine.evaluate = noise, validate
        if args.variant == "pink":
            from pink_training_noise import make_pink_for_model
            engine.make_noisy_for_model = make_pink_for_model
        elif args.variant == "nonstationary":
            from nonstationary_training_noise import noise_for_model
            engine.make_noisy_for_model = noise_for_model
        method, pod = "baseline", None
        if args.variant == "output_constraint":
            bank, _ = bank_for(args.dataset, "envelope")
            pod = PODProjector(bank["mean"], bank["basis"][:32]).to(device)
            method = "output_constraint"
        engine.train_one(args.model, method, cfg, pod, device, out)
    return {"training_output": str(out), "epochs": args.epochs}


def verify_files(directory, check_hash):
    index = directory / "SHA256SUMS"
    if not index.is_file():
        raise FileNotFoundError(f"No bundle checksum index at {index}")
    checked = 0
    for line in index.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        path = directory / relative
        if not path.is_file() or (check_hash and sha(path) != digest):
            raise ValueError(f"Missing or changed packaged file: {path}")
        checked += 1
    return checked


def verify(args):
    code_files = verify_files(ROOT, args.hash)
    if args.code_only:
        return {"code_files": code_files, "checksums_verified": args.hash}
    if not (ASSET_ROOT / "SHA256SUMS").is_file():
        raise FileNotFoundError(
            f"Asset bundle index not found at {ASSET_ROOT}. "
            "Set PAPER_ASSET_ROOT to the complete asset folder or pass --code-only."
        )
    manifest = load_json(ROOT / "manifest.json")
    for dataset, sizes in manifest["datasets"].items():
        for split, count in sizes.items():
            x, y = dataset_arrays(dataset, split)
            if x.dtype != np.float32 or y.dtype != np.int64 or x.shape != (count, 1, 4096) or y.shape != (count,):
                raise ValueError(f"Bad split: {dataset}/{split}")
    asset_files = verify_files(ASSET_ROOT, args.hash)
    return {"code_files": code_files, "asset_files": asset_files, "asset_root": str(ASSET_ROOT),
            "runs": len(manifest["checkpoints"]), "datasets": manifest["datasets"],
            "checksums_verified": args.hash}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("verify")
    check.add_argument("--hash", action="store_true")
    check.add_argument("--code-only", action="store_true", help="Check the Git-ready folder without experiment assets.")
    clean = commands.add_parser("clean")
    obs = commands.add_parser("evaluate")
    training = commands.add_parser("train")
    for command in (clean, obs, training):
        command.add_argument("--dataset", choices=DATASETS if command is clean else DATASETS[:3], required=True)
        command.add_argument("--out", type=Path, help="Optional JSON report path.")
    for command in (clean, obs):
        command.add_argument("--representation", choices=("envelope", "fft", "time"), default="envelope")
        command.add_argument("--rank", type=int, choices=RANKS, default=32)
    clean.add_argument("--all-ranks", action="store_true")
    for command in (obs, training):
        command.add_argument("--model", choices=MODELS, required=True)
        command.add_argument("--seed", type=int, choices=range(41, 46), default=42)
        command.add_argument("--variant", choices=VARIANTS, default="main")
        command.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    obs.add_argument("--snr", type=int, choices=SNRS, default=-6)
    obs.add_argument("--infer", action="store_true")
    obs.add_argument("--batch-size", type=int)
    obs.add_argument("--reduction", choices=("pod", "sparse", "kernel"), default="pod")
    training.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    run = {"verify": verify, "clean": clean_result, "evaluate": evaluate, "train": train}[args.command]
    with threadpool_limits(limits=4) if args.command in ("clean", "evaluate") else nullcontext():
        result = run(args)
    report = json.dumps(result, indent=2, ensure_ascii=False)
    print(report)
    if getattr(args, "out", None):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
