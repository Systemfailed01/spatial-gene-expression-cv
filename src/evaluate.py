import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import torch


def evaluate_encoder(encoder, head, loader, gene_names, device, encoder_name, model_path):
    """Compute per-gene Pearson correlation on a held-out test set."""
    from src.model import get_features

    head.load_state_dict(torch.load(model_path, map_location=device))
    head.eval()
    encoder.eval()

    preds, targets = [], []
    with torch.no_grad():
        for patches, expr in loader:
            patches = patches.to(device)
            features = get_features(encoder, patches, encoder_name)
            preds.append(head(features).cpu().numpy())
            targets.append(expr.numpy())

    preds   = np.concatenate(preds)
    targets = np.concatenate(targets)

    results = []
    for i, gene in enumerate(gene_names):
        r, _ = pearsonr(preds[:, i], targets[:, i])
        results.append({'gene': gene, 'pearson': r, 'encoder': encoder_name})

    df = pd.DataFrame(results)
    print(f'{encoder_name:10}  mean_pcc={df["pearson"].mean():.4f}  '
          f'max={df["pearson"].max():.4f}  min={df["pearson"].min():.4f}')
    return df, preds, targets
