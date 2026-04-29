"""
export_shape_data.py  (nam_lss_v2 version)
==========================================
從 nam_lss_v2.py 訓練好的 checkpoint 產生 shape_data.json。

使用方式：
    python export_shape_data.py --fold 1
    python export_shape_data.py --fold all
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
import os
import sys
import math
from datetime import datetime

sys.path.insert(0, '/home/iir/ryan/shared')
from nam_lss_v2 import Config, engineer_features

import pandas as pd
from joblib import load


# ============================================================
# 重建 nam_lss_v2 的模型架構
# state_dict key 格式：
#   feature_nets.i.rbf.centers / log_widths
#   feature_nets.i.mlp.{0,1,4,5}.weight/bias  (Linear+LN+GELU+Drop)*2
#   feature_nets.i.res_proj.weight/bias
#   attention.importance
#   bias (64,)
#   output_head.pi_head / alpha_head / beta_head
# ============================================================

class RBFLayer(nn.Module):
    def __init__(self, num_bases=32, input_range=(-3, 3)):
        super().__init__()
        centers = torch.linspace(input_range[0], input_range[1], num_bases)
        self.centers = nn.Parameter(centers)
        w = (input_range[1] - input_range[0]) / (num_bases - 1)
        self.log_widths = nn.Parameter(torch.full((num_bases,), math.log(w)))

    def forward(self, x):
        if x.dim() == 2:
            x = x.squeeze(-1)
        x = torch.clamp(x, -10, 10)
        widths = torch.exp(torch.clamp(self.log_widths, -5, 5)) + 0.1
        dist = (x.unsqueeze(-1) - self.centers) / widths
        return torch.exp(-0.5 * dist.clamp(-10, 10) ** 2)


class FeatureNet(nn.Module):
    def __init__(self, num_bases, hidden_dims, dropout=0.0):
        super().__init__()
        self.rbf = RBFLayer(num_bases)
        layers, prev = [], num_bases
        for hd in hidden_dims:
            layers += [nn.Linear(prev, hd), nn.LayerNorm(hd), nn.GELU(), nn.Dropout(dropout)]
            prev = hd
        self.mlp = nn.Sequential(*layers)
        if num_bases != hidden_dims[-1]:
            self.res_proj = nn.Linear(num_bases, hidden_dims[-1])
        else:
            self.res_proj = None

    def forward(self, x):
        rbf_out = self.rbf(x)
        mlp_out = self.mlp(rbf_out)
        res = self.res_proj(rbf_out) if self.res_proj else rbf_out
        return mlp_out + 0.1 * res


class MixtureBetaHead(nn.Module):
    def __init__(self, input_dim, num_components=3):
        super().__init__()
        self.K = num_components
        self.pi_head    = nn.Linear(input_dim, num_components)
        self.alpha_head = nn.Linear(input_dim, num_components)
        self.beta_head  = nn.Linear(input_dim, num_components)

    def forward(self, x):
        pi    = F.softmax(self.pi_head(x), dim=-1)
        alpha = torch.clamp(F.softplus(self.alpha_head(x)) + 1.01, 1.01, 100)
        beta  = torch.clamp(F.softplus(self.beta_head(x))  + 1.01, 1.01, 100)
        means = alpha / (alpha + beta)
        pred  = (pi * means).sum(-1)
        return torch.clamp(pred, 0.001, 0.999)


class NAMLSSV2Rebuilt(nn.Module):
    def __init__(self, num_features, num_bases, hidden_dims, num_mixture=3):
        super().__init__()
        self.num_features = num_features
        self.feature_nets = nn.ModuleList([
            FeatureNet(num_bases, hidden_dims) for _ in range(num_features)
        ])
        feat_dim = hidden_dims[-1]
        self.attention_importance = nn.Parameter(torch.zeros(num_features))
        self.bias        = nn.Parameter(torch.zeros(feat_dim))
        self.output_head = MixtureBetaHead(feat_dim, num_mixture)

    def forward(self, x):
        feat_outs = [self.feature_nets[i](x[:, i:i+1]) for i in range(self.num_features)]
        stacked = torch.stack(feat_outs, dim=1)          # (B, F, D)
        w = F.softmax(self.attention_importance, dim=-1) # (F,)
        agg = (stacked * w.unsqueeze(0).unsqueeze(-1)).sum(dim=1) + self.bias
        return self.output_head(agg)

    def load_v2_state(self, state):
        """處理 attention key 名稱差異"""
        new_state = {}
        for k, v in state.items():
            if k == 'attention.importance':
                new_state['attention_importance'] = v
            else:
                new_state[k] = v
        missing, unexpected = self.load_state_dict(new_state, strict=False)
        critical = [k for k in missing
                    if 'attention_importance' not in k]
        if critical:
            print(f"  Warning missing: {critical[:3]}")


def detect_arch(state):
    """從 state_dict 反推架構參數"""
    dims, layer = [], 0
    while f'feature_nets.0.mlp.{layer*4}.weight' in state:
        dims.append(state[f'feature_nets.0.mlp.{layer*4}.weight'].shape[0])
        layer += 1
    num_bases = state['feature_nets.0.rbf.centers'].shape[0]
    num_mixture = state['output_head.pi_head.weight'].shape[0]
    return num_bases, dims, num_mixture


def load_ensemble(model_states, num_features, device):
    state0 = model_states[0]
    num_bases, hidden_dims, num_mixture = detect_arch(state0)
    print(f"  Arch: num_bases={num_bases}, hidden_dims={hidden_dims}, mixture={num_mixture}")

    models = []
    for state in model_states:
        m = NAMLSSV2Rebuilt(num_features, num_bases, hidden_dims, num_mixture).to(device)
        m.load_v2_state(state)
        m.eval()
        models.append(m)
    return models


# ============================================================
# Shape function extraction
# ============================================================
def extract_shape_function(models, feature_idx, x_values, num_features, device):
    baseline = torch.zeros(1, num_features, device=device)
    with torch.no_grad():
        base_pred = np.mean([m(baseline).item() for m in models])

    all_contrib = []
    with torch.no_grad():
        for xi in x_values:
            x_in = baseline.clone()
            x_in[0, feature_idx] = float(xi)
            ep = [m(x_in).item() - base_pred for m in models]
            all_contrib.append(ep)

    arr = np.array(all_contrib)
    return arr.mean(axis=1).tolist(), arr.std(axis=1).tolist()


# ============================================================
# Build one fold
# ============================================================
def build_shape_data(fold_idx, checkpoint_dir, prep_path, config):
    print(f"\n{'='*60}")
    print(f"Building shape data for Fold {fold_idx}")
    print(f"{'='*60}")

    candidates = [
        os.path.join(checkpoint_dir, f'fold{fold_idx}', 'best.pt'),
        os.path.join(checkpoint_dir, f'fold{fold_idx}', 'best_checkpoint.pt'),
        os.path.join(checkpoint_dir, f'fold{fold_idx}_v2.pt'),
    ]
    ckpt_path = next((p for p in candidates if os.path.exists(p)), None)
    if not ckpt_path:
        raise FileNotFoundError(f"找不到 checkpoint，嘗試過：\n" + "\n".join(candidates))
    print(f"Loading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    model_states = ckpt['model_states']
    print(f"  Ensemble size: {len(model_states)}")

    prep      = load(prep_path)
    scaler    = prep['scaler']
    cont_cols = prep['cont_cols']

    all_features = config.BASE_FEATURES + config.ENGINEERED_FEATURES
    num_features = len(all_features)
    device = torch.device('cpu')
    models = load_ensemble(model_states, num_features, device)

    n_pts = 100
    feature_shapes = {}

    for feat_idx, fname in enumerate(all_features):
        print(f"  [{feat_idx+1:2d}/{num_features}] {fname} ...", end=' ', flush=True)
        x_transformed = np.linspace(-3, 3, n_pts)

        if fname in cont_cols:
            col_idx = cont_cols.index(fname)
            dummy = np.zeros((n_pts, len(cont_cols)))
            dummy[:, col_idx] = x_transformed
            try:
                x_original = scaler.inverse_transform(dummy)[:, col_idx].tolist()
            except Exception:
                x_original = x_transformed.tolist()
        else:
            x_original = x_transformed.tolist()

        mean_c, std_c = extract_shape_function(
            models, feat_idx, x_transformed, num_features, device
        )

        feature_shapes[fname] = {
            'feature_name':  fname,
            'feature_idx':   feat_idx,
            'x_transformed': x_transformed.tolist(),
            'x_original':    x_original,
            'y_mean':        mean_c,
            'y_std':         std_c,
            'is_binary':     fname in config.BIN_COLS,
            'deltas':        [0.0] * n_pts,
        }
        print(f"done  [{min(x_original):.1f}, {max(x_original):.1f}]")

    return feature_shapes


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', default='all')
    parser.add_argument('--out',  default='./shape_data.json')
    args = parser.parse_args()

    config = Config()
    folds  = [1, 2, 3] if args.fold == 'all' else [int(args.fold)]

    CHECKPOINT_DIR = '/home/iir/ryan/shared/Checkpoints/IECV_V2'
    SCALAR_DIR     = '/home/iir/ryan/shared/Scalar/IECV_V2'

    output = {
        'version':       'nam_shape_v2',
        'exported_at':   datetime.now().isoformat(),
        'feature_names': config.BASE_FEATURES + config.ENGINEERED_FEATURES,
        'folds':         {}
    }

    for fold_idx in folds:
        prep_path = os.path.join(SCALAR_DIR, f'fold{fold_idx}_prep.joblib')
        if not os.path.exists(prep_path):
            raise FileNotFoundError(f"找不到 scaler: {prep_path}")
        output['folds'][str(fold_idx)] = build_shape_data(
            fold_idx, CHECKPOINT_DIR, prep_path, config
        )

    os.makedirs(os.path.dirname(args.out) if os.path.dirname(args.out) else '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nExported: {args.out}  ({os.path.getsize(args.out)/1e6:.2f} MB)\n")


if __name__ == '__main__':
    main()
