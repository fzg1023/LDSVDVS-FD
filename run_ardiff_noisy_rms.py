"""100-epoch ARDiff noisy-input-RMS normalization adaptation, separate from v2."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '4')

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

import run_ardiff_reproduction as engine
from models.ardiff_equations import cosine_alpha_bar, nearest_step_for_snr
from models.ardiff_reproduction import ARDiffPaperUNet, ARDiffStepPredictor
from models.ardiff_noisy_rms import (NoisyRMSNoisePredictor, NoisyRMSStepPredictor,
                                     denoise_noisy_rms, rms_diffusion_batch)
from transforms.frequency import batch_frequency_transform
from utils.metrics import frequency_metrics, waveform_metrics
from utils.seed import set_seed

ROOT = engine.ROOT
OUT = ROOT / 'outputs/ardiff_v3_noisy_rms_100ep'


def make_models(args, device):
    return (NoisyRMSNoisePredictor(ARDiffPaperUNet(args.width, args.levels, args.heads)).to(device),
            NoisyRMSStepPredictor(ARDiffStepPredictor()).to(device))


def rms_losses(model, predictor, raw, normalizer, alpha, args, device, generator=None):
    batch = rms_diffusion_batch(raw.to(device), alpha, generator)
    with engine.autocast(device, args.amp):
        noise_in_state_units = model(batch['state'], batch['steps'])
        step_prediction = predictor(batch['state'])
    # Both the backbone's input and its noise target use RMS(corrupted state).
    diff_loss = F.mse_loss(noise_in_state_units.float()/batch['state_rms'],
                          batch['noise_target_normalized'])
    step_loss = F.l1_loss(step_prediction.float(), batch['steps'].float()/args.steps)
    return diff_loss, step_loss


@contextmanager
def rms_training_callback():
    # Reuse the checked optimizer/checkpoint engine without editing its source.
    previous = engine.losses
    engine.losses = rms_losses
    try:
        yield
    finally:
        engine.losses = previous


@torch.no_grad()
def evaluate(model, predictor, alpha, args, device, out, dataset, seed, bank):
    from run_disjoint_hust_bjtu import config_for
    from run_unet_five_seed_observation import observe
    frequency_config = config_for(dataset)['frequency']
    reference = joblib.load(bank)
    reference_hash = engine.digest(bank)
    model.eval()
    predictor.eval()
    folder = out / 'evaluation'
    folder.mkdir()
    rows = []
    for snr in engine.SNRS:
        original = engine.SOURCE/'runs'/dataset/'rescnn/seed_41'/f'signals_snr_{snr}.npz'
        with np.load(original) as archive:
            clean, noisy, labels = archive['clean'].copy(), archive['noisy'].copy(), archive['labels'].copy()
        outputs, starts, scales = [], [], []
        torch.cuda.synchronize()
        began = time.perf_counter()
        for start in range(0, len(noisy), args.eval_batch_size):
            x = torch.as_tensor(noisy[start:start+args.eval_batch_size], device=device)
            restored, steps, scale = denoise_noisy_rms(model, predictor, x, alpha,
                                                       amp=args.amp, step_prediction_scale=args.steps)
            outputs.append(restored.cpu().numpy())
            starts.append(steps.cpu().numpy())
            scales.append(scale.cpu().numpy())
            print(f'EVAL RMS {dataset}/{seed} SNR={snr}: {min(start+len(x),len(noisy))}/{len(noisy)}', flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-began
        pred, steps, noisy_rms = np.concatenate(outputs), np.concatenate(starts), np.concatenate(scales)
        spectra = []
        for array in [clean, pred]:
            chunks = []
            for start in range(0, len(array), args.eval_batch_size):
                chunks.append(batch_frequency_transform(torch.as_tensor(array[start:start+args.eval_batch_size],
                                                                         device=device),
                                                        **frequency_config).cpu().numpy())
            spectra.append(np.concatenate(chunks))
        diagnostic, fp, pp, rp, coefficients = observe(reference, spectra[1], labels)
        nominal = int(nearest_step_for_snr(float(snr), alpha).item())
        row = {'dataset': dataset, 'model': 'ARDiff-noisy-RMS-adaptation', 'seed': seed,
               'epochs': 100, 'rank': 32, 'snr_db': snr,
               **waveform_metrics(clean, pred, noisy),
               'fft_corr': frequency_metrics(np.log1p(np.abs(np.fft.rfft(clean))),
                                             np.log1p(np.abs(np.fft.rfft(pred))))['fpcc'],
               'envelope_corr': frequency_metrics(*spectra)['envelope_corr'], **diagnostic,
               'predicted_step_mean': float(steps.mean()), 'predicted_step_min': int(steps.min()),
               'predicted_step_max': int(steps.max()),
               'unit_power_nominal_step': nominal,
               'step_mae_vs_unit_power_nominal': float(np.abs(steps-nominal).mean()),
               'inference_seconds': elapsed, 'milliseconds_per_sample': elapsed*1000/len(noisy)}
        rows.append(row)
        np.savez_compressed(folder/f'signals_snr_{snr}.npz', clean=clean, noisy=noisy, denoised=pred,
                            labels=labels, input_noisy_rms=noisy_rms, predicted_steps=steps,
                            full_predictions=fp, pod_predictions=pp, random_predictions=rp, denoised_pod=coefficients)
        for name, predictions in [('full', fp), ('pod', pp)]:
            pd.DataFrame(confusion_matrix(labels, predictions, labels=np.unique(labels))).to_csv(
                folder/f'confusion_{name}_snr_{snr}.csv', index=False)
        pd.DataFrame(rows).to_csv(out/'metrics_by_snr.csv', index=False, encoding='utf-8-sig')
    if reference_hash != engine.digest(bank):
        raise ValueError('Frozen observation reference changed')
    return rows


def smoke(args, device):
    set_seed(42)
    model, predictor = make_models(args, device)
    alpha = cosine_alpha_bar(args.steps, offset=args.cosine_offset, device=device, dtype=torch.float32)
    raw = torch.randn(args.smoke_batch_size, 1, 4096, device=device)
    raw *= torch.logspace(-2, 1, len(raw), device=device).view(-1, 1, 1)
    ld, lp = rms_losses(model, predictor, raw, None, alpha, args, device)
    (ld+lp).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for net in [model, predictor] for p in net.parameters())
    model.eval()
    predictor.eval()
    restored, steps, scale = denoise_noisy_rms(model, predictor, raw, alpha, start_steps=2,
                                               amp=args.amp, step_prediction_scale=args.steps)
    assert restored.shape == raw.shape and torch.isfinite(restored).all() and (steps == 2).all()
    result = {'batch': len(raw), 'length': 4096, 'finite_gradients': True, 'two_step_reverse': True,
              'diffusion_parameters': sum(p.numel() for p in model.parameters()),
              'nspm_parameters': sum(p.numel() for p in predictor.parameters()),
              'normalized_noise_mse': float(ld.detach()), 'step_mae': float(lp.detach())*args.steps}
    args.output.mkdir(parents=True, exist_ok=True)
    engine.save_json(args.output/'smoke_check.json', result)
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['hust', 'bjtu_rao', 'both'], default='both')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42])
    parser.add_argument('--epochs', type=int, choices=[100], default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--micro-batch-size', type=int, default=32)
    parser.add_argument('--validation-batch-size', type=int, default=16)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--cosine-offset', type=float, default=.008)
    parser.add_argument('--width', type=int, default=32)
    parser.add_argument('--levels', type=int, default=3)
    parser.add_argument('--heads', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--smoke-batch-size', type=int, default=32)
    args = parser.parse_args()
    args.step_target = 'normalized'
    if min(args.batch_size, args.micro_batch_size, args.smoke_batch_size) < 2 or args.steps < 3:
        parser.error('Invalid batch sizes or diffusion step count')
    if min(args.validation_batch_size, args.eval_batch_size) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Invalid evaluation settings or duplicate seeds')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    device = torch.device('cuda')
    if args.smoke_only:
        smoke(args, device)
        return
    datasets = ['hust', 'bjtu_rao'] if args.dataset == 'both' else [args.dataset]
    all_rows = []
    for dataset in datasets:
        arrays, protocol, bank = engine.load_data(dataset)
        for seed in args.seeds:
            out = args.output/dataset/f'seed_{seed}'
            out.mkdir(parents=True, exist_ok=False)
            set_seed(seed)
            model, predictor = make_models(args, device)
            alpha = cosine_alpha_bar(args.steps, offset=args.cosine_offset, device=device, dtype=torch.float32)
            plan = {**vars(args), 'output': str(args.output), 'dataset': dataset, 'seed': seed,
                    'model': 'ARDiff noisy-input RMS adaptation',
                    'parameters': {'diffusion': sum(p.numel() for p in model.parameters()),
                                   'nspm': sum(p.numel() for p in predictor.parameters())},
                    'data_sha256': protocol['array_sha256'], 'reference_sha256': engine.digest(bank),
                    'normalization': {'outer': 'observed noisy RMS only; multiply back after restoration',
                        'training_reference': 'clean RMS defines unit-power forward synthesis, not input scaling',
                        'backbone_input': 'noisy/current diffusion state divided by its own RMS',
                        'targets': 'corresponding clean and noise targets use the same corrupted-state scale',
                        'noise_output': 'multiply normalized noise by current-state RMS before reverse update'},
                    'loss': 'normalized-coordinate epsilon MSE + MAE(t/T); no class or POD loss',
                    'scope': 'Normalization adaptation, not exact original-paper or a scalar-only MinMax ablation',
                    'vsm_boundary': 'Original VSM call retained. Positive scale-equivariance makes initial '
                                    'VSM gain cancel with its inverse in exact arithmetic; not a VSM benefit test.',
                    'inherited_architecture_choices': ['width/depth/head count selected by project',
                        'three equal-width rFFT bands; channel-wise LayerNorm; one outer FAM residual',
                        'NSPM global average pooling',
                        'standard cosine offset/beta cap; not a reconstruction of paper Table 5'],
                    'reference': 'same clean-train frozen Full/POD/Random, rank32',
                    'repetitions': 'one seed initially; not five-fold CV'}
            engine.save_json(out/'experiment_plan.json', plan)
            relatives = ['run_ardiff_noisy_rms.py', 'models/ardiff_noisy_rms.py',
                         'run_ardiff_reproduction.py', 'models/ardiff_reproduction.py',
                         'models/ardiff_equations.py', 'models/ardiff_inspired.py']
            for relative in relatives:
                target = out/'source_snapshot'/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT/relative, target)
            engine.save_json(out/'source_sha256.json', {r: engine.digest(ROOT/r) for r in relatives})
            began = time.perf_counter()
            with rms_training_callback():
                selected = engine.run_training(model, predictor, arrays, None, alpha, args, device, out, plan)
            engine.save_json(out/'status.json', {'stage': 'evaluation', 'epochs': 100})
            all_rows.extend(evaluate(model, predictor, alpha, args, device, out, dataset, seed, bank))
            engine.save_json(out/'complete.json', {'epochs': 100, 'best_epochs': selected,
                'elapsed_seconds': time.perf_counter()-began,
                'weights_sha256': {name: engine.digest(out/name)
                                  for name in ['best_diffusion.pth', 'best_step_predictor.pth']}})
            engine.save_json(out/'status.json', {'stage': 'complete', 'epochs': 100})
            pd.DataFrame(all_rows).to_csv(args.output/'all_runs_by_snr.csv', index=False, encoding='utf-8-sig')
    summary = pd.DataFrame(all_rows)
    for field in ['Full Logistic', 'POD Logistic', 'Random Logistic', 'Full Macro F1', 'POD Macro F1', 'Random Macro F1']:
        summary[field] *= 100
    summary.to_csv(args.output/'results_percent.csv', index=False, encoding='utf-8-sig')
    print(summary[summary.snr_db == -6].to_string(index=False), flush=True)
    print('ALL NOISY-RMS RUNS COMPLETE', flush=True)


if __name__ == '__main__':
    main()
