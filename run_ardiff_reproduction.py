"""Run the independently reconstructed ARDiff on the existing disjoint splits.

No action is taken on import. --smoke-only does not train or evaluate a full run.
New results never overwrite the previous ARDiff-inspired experiments.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from models.ardiff_equations import cosine_alpha_bar, nearest_step_for_snr
from models.ardiff_reproduction import (ARDiffPaperUNet, ARDiffStepPredictor,
    TrainMinMax, forward_sample, reverse_denoise)
from transforms.frequency import batch_frequency_transform
from utils.metrics import frequency_metrics, waveform_metrics
from utils.seed import set_seed

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'outputs/disjoint_hust_bjtu_two_baselines_five_seed_100ep_rank32'
DEFAULT_OUT = ROOT / 'outputs/ardiff_paper_v2_stepnorm_batch32_100ep'
SNRS = [0, -2, -4, -6]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def slices_without_singleton(count, batch):
    if count < 2 or batch < 2:
        raise ValueError('BatchNorm training requires at least two examples')
    if batch == 2 and count % 2:
        raise ValueError('An odd sample count cannot be partitioned into batches of two')
    start = 0
    while start < count:
        size = min(batch, count-start)
        if count-start-size == 1:
            size -= 1
        yield slice(start, start+size)
        start += size


def autocast(device, amp):
    return torch.autocast(device.type, dtype=torch.float16, enabled=amp and device.type == 'cuda')


def load_data(dataset):
    # Reuse the existing raw-time disjoint protocol without generating another split.
    folder = SOURCE / 'data' / dataset
    protocol = json.loads((folder / 'data_protocol.json').read_text(encoding='utf-8'))
    for relative, expected in protocol['array_sha256'].items():
        if digest(folder / relative) != expected:
            raise ValueError(f'Data hash mismatch: {folder / relative}')
    arrays = {}
    for split in ['train', 'val', 'test']:
        array = np.load(folder / split / 'x.npy')
        labels = np.load(folder / split / 'y.npy')
        if array.ndim != 3 or array.shape[1:] != (1, 4096) or not np.isfinite(array).all():
            raise ValueError(f'Invalid {dataset}/{split} waveform array')
        if len(array) != len(labels):
            raise ValueError('Signal/label length mismatch')
        arrays[split] = torch.from_numpy(array).float()
    bank = SOURCE / 'reference' / dataset / 'clean_classifiers.joblib'
    if not bank.is_file():
        raise FileNotFoundError(f'Existing frozen observation bank required: {bank}')
    return arrays, protocol, bank


def make_models(args, device):
    return (ARDiffPaperUNet(args.width, args.levels, args.heads).to(device),
            ARDiffStepPredictor().to(device))


def step_scale(args):
    return args.steps if getattr(args, 'step_target', 'raw') == 'normalized' else 1


def losses(model, predictor, raw, normalizer, alpha, args, device, generator=None):
    clean = normalizer.encode(raw.to(device))
    xt, steps, noise = forward_sample(clean, alpha, generator)
    with autocast(device, args.amp):
        noise_prediction = model(xt, steps)
        step_prediction = predictor(xt)
    return (F.mse_loss(noise_prediction.float(), noise),
            F.l1_loss(step_prediction.float(), steps.float() / step_scale(args)))


@torch.no_grad()
def validate(model, predictor, val, normalizer, alpha, args, device):
    model.eval()
    predictor.eval()
    generator = torch.Generator(device=device).manual_seed(78123)
    sums = np.zeros(2)
    val_batch = getattr(args, 'validation_batch_size', args.micro_batch_size)
    for start in range(0, len(val), val_batch):
        raw = val[start:start+val_batch]
        ld, lp = losses(model, predictor, raw, normalizer, alpha, args, device, generator)
        sums += np.array([float(ld), float(lp)*step_scale(args)]) * len(raw)
    return (sums / len(val)).tolist()


def run_training(model, predictor, arrays, normalizer, alpha, args, device, out, plan):
    optimizers = [torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
                  for net in [model, predictor]]
    schedulers = [torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs) for opt in optimizers]
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp and device.type == 'cuda')
    updates = [0, 0]
    hooks = []
    for index, opt in enumerate(optimizers):
        def counted_step(optimizer, positional, keyword, index=index):
            updates[index] += 1
        hooks.append(opt.register_step_post_hook(counted_step))
    best = [float('inf'), float('inf')]
    history = []
    for epoch in range(1, args.epochs+1):
        model.train()
        predictor.train()
        order = torch.randperm(len(arrays['train']))
        sums = np.zeros(2)
        for group in slices_without_singleton(len(order), args.batch_size):
            indices = order[group]
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            for micro in slices_without_singleton(len(indices), args.micro_batch_size):
                raw = arrays['train'][indices[micro]]
                ld, lp = losses(model, predictor, raw, normalizer, alpha, args, device)
                if not torch.isfinite(ld+lp):
                    raise ValueError('Nonfinite training loss')
                # Networks have disjoint parameters; summing losses does not share gradients.
                scaler.scale((ld+lp) * (len(raw)/len(indices))).backward()
                sums += np.array([float(ld.detach()), float(lp.detach())*step_scale(args)]) * len(raw)
            for net, opt in zip([model, predictor], optimizers):
                scaler.unscale_(opt)
                # Float16 overflow is handled by GradScaler's skipped step/backoff.
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0, error_if_nonfinite=not scaler.is_enabled())
                scaler.step(opt)
            scaler.update()
        vd, vp = validate(model, predictor, arrays['val'], normalizer, alpha, args, device)
        if not np.isfinite([vd, vp]).all():
            raise ValueError('Nonfinite validation loss')
        for i, (net, value, name) in enumerate(zip([model, predictor], [vd, vp],
                                                ['best_diffusion.pth', 'best_step_predictor.pth'])):
            if value < best[i]:
                best[i] = value
                torch.save({'model': net.state_dict(), 'epoch': epoch, 'validation_loss': value,
                            'plan': plan}, out / name)
        row = {'epoch': epoch, 'train_diff_mse': sums[0]/len(order), 'train_step_mae': sums[1]/len(order),
               'val_diff_mse': vd, 'val_step_mae': vp, 'lr': optimizers[0].param_groups[0]['lr'],
               'train_step_loss': sums[1]/len(order)/step_scale(args),
               'val_step_loss': vp/step_scale(args), 'step_prediction_scale': step_scale(args),
               'diff_optimizer_steps': updates[0], 'nspm_optimizer_steps': updates[1]}
        history.append(row)
        pd.DataFrame(history).to_csv(out / 'train_val_log.csv', index=False)
        save_json(out / 'status.json', {'stage': 'training', 'epoch': epoch, 'epochs': args.epochs})
        for scheduler in schedulers:
            scheduler.step()
        print(f'{out.parent.name}/{out.name} epoch {epoch}/{args.epochs}: '
              f'diff={row["train_diff_mse"]:.5f}, step MAE={row["train_step_mae"]:.2f}, '
              f'val={vd:.5f}/{vp:.2f}', flush=True)
    torch.save({'diffusion': model.state_dict(), 'predictor': predictor.state_dict(),
                'epoch': args.epochs, 'plan': plan}, out / 'last_epoch.pth')
    save_json(out / 'optimizer_updates.json', {'diffusion': updates[0], 'nspm': updates[1],
              'definition': 'Actual optimizer.step calls after AMP overflow handling'})
    for hook in hooks:
        hook.remove()
    selected = []
    for net, name in zip([model, predictor], ['best_diffusion.pth', 'best_step_predictor.pth']):
        checkpoint = torch.load(out / name, map_location=device, weights_only=False)
        net.load_state_dict(checkpoint['model'])
        selected.append(checkpoint['epoch'])
    return selected


@torch.no_grad()
def evaluate(model, predictor, normalizer, alpha, args, device, out, dataset, seed, bank):
    from run_disjoint_hust_bjtu import config_for
    from run_unet_five_seed_observation import observe
    frequency_config = config_for(dataset)['frequency']
    reference = joblib.load(bank)
    before = digest(bank)
    original = SOURCE / 'runs' / dataset / 'rescnn' / 'seed_41'
    model.eval()
    predictor.eval()
    target = out / 'evaluation'
    target.mkdir()
    rows = []
    for snr in SNRS:
        with np.load(original / f'signals_snr_{snr}.npz') as archive:
            clean, noisy, labels = archive['clean'].copy(), archive['noisy'].copy(), archive['labels'].copy()
        if clean.shape != noisy.shape or clean.shape[1:] != (1, 4096):
            raise ValueError('Unexpected cached test waveforms')
        predictions, starts = [], []
        if device.type == 'cuda':
            torch.cuda.synchronize()
        began = time.perf_counter()
        for i in range(0, len(noisy), args.eval_batch_size):
            raw = torch.as_tensor(noisy[i:i+args.eval_batch_size], device=device)
            y, step = reverse_denoise(model, predictor, normalizer.encode(raw), alpha, amp=args.amp,
                                     step_prediction_scale=step_scale(args))
            predictions.append(normalizer.decode(y).cpu().numpy())
            starts.append(step.cpu().numpy())
            print(f'evaluation {dataset}/{seed} SNR={snr}: {min(i+len(raw),len(noisy))}/{len(noisy)}', flush=True)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        inference_seconds = time.perf_counter()-began
        pred, steps = np.concatenate(predictions), np.concatenate(starts)
        spectra = []
        for array in [clean, pred]:
            spectra.append(np.concatenate([
                batch_frequency_transform(torch.as_tensor(array[i:i+args.eval_batch_size], device=device),
                                          **frequency_config).cpu().numpy()
                for i in range(0, len(array), args.eval_batch_size)]))
        diagnostics, fp, pp, rp, coefficients = observe(reference, spectra[1], labels)
        row = {'dataset': dataset, 'model': 'ARDiff-paper-reimplementation-v2', 'seed': seed, 'snr_db': snr,
               'epochs': args.epochs, 'rank': 32, 'step_target': args.step_target,
               **waveform_metrics(clean, pred, noisy),
               'fft_corr': frequency_metrics(np.log1p(np.abs(np.fft.rfft(clean))),
                                             np.log1p(np.abs(np.fft.rfft(pred))))['fpcc'],
               'envelope_corr': frequency_metrics(*spectra)['envelope_corr'], **diagnostics,
               'predicted_step_mean': float(steps.mean()), 'inference_seconds': inference_seconds,
               'milliseconds_per_sample': inference_seconds*1000/len(clean)}
        rows.append(row)
        np.savez_compressed(target / f'signals_snr_{snr}.npz', clean=clean, noisy=noisy, denoised=pred,
                            labels=labels, predicted_steps=steps, full_predictions=fp, pod_predictions=pp,
                            random_predictions=rp, denoised_pod=coefficients)
        pd.DataFrame(rows).to_csv(out / 'metrics_by_snr.csv', index=False, encoding='utf-8-sig')
    if digest(bank) != before:
        raise ValueError('Frozen clean observer changed')
    return rows


def assumptions():
    return {
        'identity': 'Independent paper-based implementation, not author source or verified metric reproduction',
        'fixed': ['encoder/decoder Res-FAM-Res-TimeAttention', 'FAM output LayerNorm',
                  'NSPM FC BatchNorm', 'full temporal attention without pooling', 'direct timestep regression'],
        'unreported_choices': ['width/depth/head count configurable; one stage per scale',
            'three equal-width rFFT bands; head dimensions partition channels',
            'LayerNorm over channels; one outer FAM residual', 'NSPM global average pooling',
            'train-only dataset-global Min-Max affine, no heldout clipping',
            'NSPM trained on forward-diffused examples; scalar target units set by step_target',
            'final sqrt(alpha_start)/VSM_scale gain, then inverse affine',
            'AdamW and cosine learning-rate decay, validation checkpoint selection',
            'microbatch BatchNorm statistics differ from a single paper-size batch'],
        'schedule': 'Standard cosine; does not reproduce paper Table 5. No post-hoc fitting to table.',
        'test_protocol': 'Same saved noisy test inputs and frozen rank32 observers as other networks',
        'no_leakage': 'Normalizer, reference space and classifiers fitted on clean train only; no labels in denoiser training'}


def smoke(args, device):
    set_seed(42)
    model, predictor = make_models(args, device)
    alpha = cosine_alpha_bar(args.steps, offset=args.cosine_offset, device=device, dtype=torch.float32)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    raw = torch.randn(args.smoke_batch_size, 1, 4096, device=device)
    normalizer = TrainMinMax.fit(raw)
    ld, lp = losses(model, predictor, raw, normalizer, alpha, args, device)
    (ld+lp).backward()
    if not all(p.grad is not None and torch.isfinite(p.grad).all()
               for net in [model, predictor] for p in net.parameters()):
        raise AssertionError('Missing or nonfinite gradient')
    model.eval()
    predictor.eval()
    y, steps = reverse_denoise(model, predictor, normalizer.encode(raw), alpha,
                              start_steps=2, amp=args.amp, step_prediction_scale=step_scale(args))
    assert y.shape == raw.shape and torch.isfinite(y).all() and (steps == 2).all()
    result = {'device': str(device), 'length': 4096, 'batch': len(raw), 'finite_gradients': True,
              'two_reverse_steps': True, 'diffusion_parameters': sum(p.numel() for p in model.parameters()),
              'nspm_parameters': sum(p.numel() for p in predictor.parameters()),
              'diffusion_loss': float(ld.detach()), 'step_loss': float(lp.detach()),
              'step_mae': float(lp.detach())*step_scale(args), 'step_target': args.step_target,
              'peak_cuda_allocated_gb': torch.cuda.max_memory_allocated()/1024**3 if device.type == 'cuda' else None}
    args.output.mkdir(parents=True, exist_ok=True)
    save_json(args.output / 'smoke_check.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['hust', 'bjtu_rao', 'both'], default='both')
    parser.add_argument('--seeds', nargs='+', type=int, default=[41, 42, 43, 44, 45])
    parser.add_argument('--epochs', type=int, choices=[100], default=100,
                        help='Fixed to 100 epochs for the current comparison protocol')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--micro-batch-size', type=int, default=32)
    parser.add_argument('--validation-batch-size', type=int, default=16,
                        help='Keep v1 validation noise and timestep RNG sequence unchanged')
    parser.add_argument('--step-target', choices=['raw', 'normalized'], default='normalized')
    parser.add_argument('--eval-batch-size', type=int, default=4)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--levels', type=int, default=3)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--cosine-offset', type=float, default=0.008)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--smoke-batch-size', type=int, default=2)
    args = parser.parse_args()
    if (args.steps < 3 or args.epochs < 1 or args.batch_size < 2 or args.micro_batch_size < 2
            or args.eval_batch_size < 1 or args.validation_batch_size < 1 or args.smoke_batch_size < 2
            or len(set(args.seeds)) != len(args.seeds)):
        parser.error('Invalid steps/epochs/batch settings or duplicate seeds')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; specify --device cpu for small tests')
    torch.set_num_threads(4)
    device = torch.device(args.device)
    if args.smoke_only:
        smoke(args, device)
        return
    datasets = ['hust', 'bjtu_rao'] if args.dataset == 'both' else [args.dataset]
    collected = []
    for dataset in datasets:
        arrays, protocol, bank = load_data(dataset)
        normalizer = TrainMinMax.fit(arrays['train'])
        for seed in args.seeds:
            out = args.output / dataset / f'seed_{seed}'
            out.mkdir(parents=True, exist_ok=False)
            set_seed(seed)
            model, predictor = make_models(args, device)
            alpha = cosine_alpha_bar(args.steps, offset=args.cosine_offset, device=device, dtype=torch.float32)
            plan = {**vars(args), 'output': str(args.output), 'dataset': dataset, 'seed': seed,
                    'normalizer': asdict(normalizer), 'data_sha256': protocol['array_sha256'],
                    'step_prediction_scale': step_scale(args),
                    'step_head': 'Unbounded linear scalar; decoded by multiplying step_prediction_scale',
                    'reference_sha256': digest(bank), 'assumptions': assumptions(),
                    'parameters': {'diffusion': sum(p.numel() for p in model.parameters()),
                                   'nspm': sum(p.numel() for p in predictor.parameters())},
                    'paper': '10.1016/j.ymssp.2025.113726'}
            save_json(out / 'experiment_plan.json', plan)
            snapshot = out / 'source_snapshot'
            relatives = ['run_ardiff_reproduction.py', 'models/ardiff_reproduction.py',
                         'models/ardiff_equations.py', 'models/ardiff_inspired.py']
            for relative in relatives:
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / relative, target)
            save_json(out / 'source_sha256.json', {r: digest(ROOT / r) for r in relatives})
            schedule_steps = nearest_step_for_snr(torch.tensor([0., -4., -8., -12.], device=device), alpha)
            save_json(out / 'schedule_check.json', {'snr_db': [0, -4, -8, -12],
                      'current_steps': schedule_steps.cpu().tolist(), 'paper_table5': [465, 616, 741, 832]})
            began = time.perf_counter()
            best_epochs = run_training(model, predictor, arrays, normalizer, alpha, args, device, out, plan)
            save_json(out / 'status.json', {'stage': 'evaluation'})
            rows = evaluate(model, predictor, normalizer, alpha, args, device, out, dataset, seed, bank)
            collected.extend(rows)
            save_json(out / 'complete.json', {'epochs': args.epochs, 'best_epochs': best_epochs,
                      'elapsed_seconds': time.perf_counter()-began,
                      'weights_sha256': {n: digest(out/n) for n in ['best_diffusion.pth', 'best_step_predictor.pth']}})
            save_json(out / 'status.json', {'stage': 'complete'})
            pd.DataFrame(collected).to_csv(args.output / 'all_runs_by_snr.csv', index=False, encoding='utf-8-sig')
    frame = pd.DataFrame(collected)
    metrics = ['snr_improvement', 'rmse', 'fft_corr', 'envelope_corr',
               'Full Logistic', 'POD Logistic', 'Random Logistic', 'milliseconds_per_sample']
    summary = frame.groupby(['dataset', 'model', 'snr_db'])[metrics].agg(['mean', 'std', 'count'])
    summary.columns = ['_'.join(column) for column in summary.columns]
    summary.to_csv(args.output / 'summary_by_snr.csv', encoding='utf-8-sig')


if __name__ == '__main__':
    main()
