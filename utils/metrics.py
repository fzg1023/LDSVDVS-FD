import numpy as np


def separability_metrics(features, labels):
    try:
        from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
    except Exception as exc:
        return {"sklearn_available": False, "error": str(exc)}
    features = np.asarray(features)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2 or len(features) <= len(np.unique(labels)):
        return {"sklearn_available": True, "silhouette": np.nan, "davies_bouldin": np.nan, "calinski_harabasz": np.nan}
    return {
        "sklearn_available": True,
        "silhouette": float(silhouette_score(features, labels)),
        "davies_bouldin": float(davies_bouldin_score(features, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(features, labels)),
    }


def waveform_metrics(clean, pred, noisy=None):
    clean = np.asarray(clean).reshape(len(clean), -1)
    pred = np.asarray(pred).reshape(len(pred), -1)
    noisy = pred if noisy is None else np.asarray(noisy).reshape(len(noisy), -1)
    eps = 1e-12
    err = pred - clean
    noisy_err = noisy - clean
    signal_power = np.mean(clean**2, axis=1) + eps
    input_snr = 10 * np.log10(signal_power / (np.mean(noisy_err**2, axis=1) + eps))
    output_snr = 10 * np.log10(signal_power / (np.mean(err**2, axis=1) + eps))
    corr = []
    for a, b in zip(clean, pred):
        corr.append(np.nan if np.std(a) < eps or np.std(b) < eps else np.corrcoef(a, b)[0, 1])
    return {
        "input_snr": float(np.mean(input_snr)),
        "output_snr": float(np.mean(output_snr)),
        "snr_improvement": float(np.mean(output_snr - input_snr)),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "pearson": float(np.nanmean(corr)),
    }


def frequency_metrics(clean_spec, pred_spec):
    clean = np.asarray(clean_spec).reshape(len(clean_spec), -1)
    pred = np.asarray(pred_spec).reshape(len(pred_spec), -1)
    eps = 1e-12
    corrs = []
    for a, b in zip(clean, pred):
        corrs.append(np.nan if np.std(a) < eps or np.std(b) < eps else np.corrcoef(a, b)[0, 1])
    return {
        "fpcc": float(np.nanmean(corrs)),
        "envelope_corr": float(np.nanmean(corrs)),
        "spectral_rmse": float(np.sqrt(np.mean((pred - clean) ** 2))),
    }


def diagnosis_metrics(train_x, train_y, test_x, test_y):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return {"error": str(exc)}, None
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5))
    knn.fit(train_x, train_y)
    kp = knn.predict(test_x)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, solver="lbfgs"))
    clf.fit(train_x, train_y)
    lp = clf.predict(test_x)
    return {
        "knn_accuracy": float(accuracy_score(test_y, kp)),
        "knn_macro_f1": float(f1_score(test_y, kp, average="macro")),
        "logistic_accuracy": float(accuracy_score(test_y, lp)),
        "logistic_macro_f1": float(f1_score(test_y, lp, average="macro")),
    }, confusion_matrix(test_y, lp)
