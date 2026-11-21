"""Feature extraction and observer fitting from the paper evaluation."""
import warnings
import numpy as np
from scipy import signal
from scipy.linalg import svd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
RANKS = [4, 8, 16, 32, 64, 128]
RANDOM_REPEATS = 10
RANDOM_SEED = 20261031

def as_signals(x):
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 2 or x.shape[1] != 4096 or not np.isfinite(x).all():
        raise ValueError('Expected finite [N,4096] or [N,1,4096] signals')
    return x.astype(np.float64)


def features(x, representation, fs):
    x = as_signals(x)
    if representation == 'time':
        values = x
    elif representation == 'fft':
        values = np.log1p(np.abs(np.fft.rfft(x, axis=-1)))
    elif representation == 'welch_psd':
        _, psd = signal.welch(x, fs=fs, window='hann', nperseg=1024, noverlap=512,
                              nfft=1024, detrend='constant', scaling='density', average='mean', axis=-1)
        floor = np.maximum(psd.max(axis=-1, keepdims=True) * 1e-8, np.finfo(np.float64).tiny)
        values = np.log(np.maximum(psd, floor))
    elif representation in ['envelope', 'squared_envelope', 'log_envelope']:
        envelope = np.abs(signal.hilbert(x, axis=-1))
        if representation == 'squared_envelope':
            envelope = envelope ** 2
        elif representation == 'log_envelope':
            floor = np.maximum(np.sqrt(np.mean(envelope ** 2, axis=-1, keepdims=True)) * 1e-8,
                               np.finfo(np.float64).tiny)
            envelope = np.log(np.maximum(envelope, floor))
            envelope = envelope - envelope.mean(axis=-1, keepdims=True)
        values = np.log1p(np.abs(np.fft.rfft(envelope, axis=-1)))
    else:
        raise ValueError(f'Unknown representation: {representation}')
    # Every representation receives the same within-sample feature standardization.
    values = (values - values.mean(axis=-1, keepdims=True)) / np.maximum(
        values.std(axis=-1, keepdims=True, ddof=1), 1e-8)
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite features')
    return result


def random_pool(dimension):
    matrices = []
    for repeat in range(RANDOM_REPEATS):
        rng = np.random.default_rng(np.random.SeedSequence([RANDOM_SEED, dimension, repeat]))
        q, _ = np.linalg.qr(rng.standard_normal((dimension, max(RANKS))), mode='reduced')
        matrices.append(q)
    return np.stack(matrices)


def fit_classifier(x, y):
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=1.0, max_iter=5000, solver='lbfgs', n_jobs=1, random_state=RANDOM_SEED))
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model.fit(x, y)
    return model


def fit_bank(train, labels, pool, info):
    train = np.asarray(train, dtype=np.float64)
    mean = train.mean(axis=0)
    centered = train - mean
    _, singular, vt = svd(centered, full_matrices=False, check_finite=True, lapack_driver='gesdd')
    bank = {'info': info, 'mean': mean, 'basis': vt[:max(RANKS)],
            'energy': np.cumsum(singular ** 2) / np.sum(singular ** 2),
            'classes': np.unique(labels), 'full': fit_classifier(train, labels), 'pod': {}, 'random': {}}
    coordinates = centered @ bank['basis'].T
    random_coordinates = [centered @ matrix for matrix in pool]
    for rank in RANKS:
        bank['pod'][rank] = fit_classifier(coordinates[:, :rank], labels)
        bank['random'][rank] = [fit_classifier(values[:, :rank], labels) for values in random_coordinates]
        print(f'FIT {info["dataset"]}/{info["representation"]} rank={rank}', flush=True)
    return bank


def predict_bank(bank, pool, test):
    centered = np.asarray(test, dtype=np.float64) - bank['mean']
    pod_coordinates = centered @ bank['basis'].T
    random_coordinates = [centered @ matrix for matrix in pool]
    return {'full': bank['full'].predict(test),
            'pod': np.stack([bank['pod'][rank].predict(pod_coordinates[:, :rank]) for rank in RANKS]),
            'random': np.stack([np.stack([clf.predict(z[:, :rank]) for clf, z in
                       zip(bank['random'][rank], random_coordinates)]) for rank in RANKS])}


def score_predictions(predictions, labels, bank, info):
    full_accuracy = accuracy_score(labels, predictions['full'])
    full_f1 = f1_score(labels, predictions['full'], average='macro', zero_division=0)
    rows = []
    for i, rank in enumerate(RANKS):
        random_scores = [accuracy_score(labels, p) for p in predictions['random'][i]]
        row = {**info, 'rank': rank, 'feature_dim': len(bank['mean']), 'test_samples': len(labels),
            'rank_fraction': rank / len(bank['mean']), 'train_cumulative_energy': bank['energy'][rank - 1],
            'full_accuracy': full_accuracy, 'full_macro_f1': full_f1,
            'pod_accuracy': accuracy_score(labels, predictions['pod'][i]),
            'pod_macro_f1': f1_score(labels, predictions['pod'][i], average='macro', zero_division=0),
            'random_accuracy': float(np.mean(random_scores)),
            'random_basis_accuracy_std': float(np.std(random_scores, ddof=1)),
            'random_macro_f1': float(np.mean([f1_score(labels, p, average='macro', zero_division=0)
                                            for p in predictions['random'][i]]))}
        row['pod_minus_full'] = row['pod_accuracy'] - row['full_accuracy']
        row['pod_minus_random'] = row['pod_accuracy'] - row['random_accuracy']
        rows.append(row)
    return rows

