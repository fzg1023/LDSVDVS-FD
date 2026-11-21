import torch
import torch.nn.functional as F


def latent_alignment_loss(z_hat, z_clean):
    return F.mse_loss(z_hat, z_clean)


def supervised_contrastive_loss(z, labels, temperature=0.2):
    if labels is None or z.shape[0] < 2:
        return z.new_tensor(0.0)
    labels = labels.view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(z.device)
    logits = torch.matmul(F.normalize(z, dim=1), F.normalize(z, dim=1).T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    eye = torch.eye(z.shape[0], device=z.device)
    mask = mask * (1.0 - eye)
    exp_logits = torch.exp(logits) * (1.0 - eye)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-8))
    positives = mask.sum(dim=1)
    valid = positives > 0
    if not torch.any(valid):
        return z.new_tensor(0.0)
    return -(mask[valid] * log_prob[valid]).sum(dim=1).div(positives[valid]).mean()


def graph_manifold_loss(z_hat, z_clean, k=5):
    n = z_clean.shape[0]
    if n <= 2:
        return z_clean.new_tensor(0.0)
    k = min(k, n - 1)
    with torch.no_grad():
        dist = torch.cdist(z_clean, z_clean)
        nn = dist.topk(k + 1, largest=False).indices[:, 1:]
    src = torch.arange(n, device=z_clean.device).view(-1, 1).expand(-1, k).reshape(-1)
    dst = nn.reshape(-1)
    clean_diff = z_clean[src] - z_clean[dst]
    hat_diff = z_hat[src] - z_hat[dst]
    return F.mse_loss(hat_diff, clean_diff)


def soft_prototype_loss(z_hat, prototypes, temperature=0.5):
    if prototypes is None or prototypes.numel() == 0:
        return z_hat.new_tensor(0.0)
    dist = torch.cdist(z_hat, prototypes).pow(2)
    q = torch.softmax(-dist / temperature, dim=1)
    z_proto = q @ prototypes
    return F.mse_loss(z_hat, z_proto)


def inter_intra_distance_metrics(features, labels):
    import numpy as np

    x = np.asarray(features)
    y = np.asarray(labels)
    rows = []
    centers = []
    classes = np.unique(y)
    for c in classes:
        xc = x[y == c]
        center = xc.mean(axis=0)
        centers.append(center)
        rows.append(np.linalg.norm(xc - center, axis=1).mean())
    centers = np.stack(centers)
    cd = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
    inter = cd[np.triu_indices_from(cd, k=1)].mean()
    intra = float(np.mean(rows))
    return {"intra_class_distance": intra, "inter_class_distance": float(inter), "inter_intra_ratio": float(inter / (intra + 1e-12))}
