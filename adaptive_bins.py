"""
adaptive_bins.py
================
Post-hoc: 根據 shape function 的梯度決定 bin 位置。
變化大的區間切更細，變化小的區間切更粗。

使用方式：
    python adaptive_bins.py --input shape_data.json --output shape_data_adaptive.json
"""

import json
import numpy as np
import argparse


def compute_adaptive_bins(y_mean, n_bins=20, min_bins=5, max_bins=40):
    """
    根據 y_mean 的梯度決定 bin 邊界。
    梯度大的區間 → 更多 bins
    梯度小的區間 → 更少 bins
    回傳 bin 邊界的 index 列表
    """
    y = np.array(y_mean)
    grad = np.abs(np.gradient(y))

    # 正規化梯度為密度權重
    grad = grad - grad.min()
    if grad.max() > 0:
        grad = grad / grad.max()
    density = grad + 0.1  # 避免完全為零的區間沒有 bin

    # 用累積密度決定 bin 邊界
    cumsum = np.cumsum(density)
    cumsum = cumsum / cumsum[-1]

    # 在累積密度上均勻取樣 n_bins 個切點
    thresholds = np.linspace(0, 1, n_bins + 1)[1:-1]
    bin_edges_idx = [0]
    for th in thresholds:
        idx = np.searchsorted(cumsum, th)
        idx = int(np.clip(idx, 1, len(y) - 1))
        if idx != bin_edges_idx[-1]:
            bin_edges_idx.append(idx)
    bin_edges_idx.append(len(y) - 1)

    return bin_edges_idx


def build_adaptive_feature(fdata, n_bins=20):
    """
    為單一特徵建立 adaptive bin 結構。
    回傳新的 bin 資訊供 UI 使用。
    """
    edges_idx = compute_adaptive_bins(fdata['y_mean'], n_bins)
    n_actual  = len(edges_idx) - 1

    bins = []
    for i in range(n_actual):
        lo_idx = edges_idx[i]
        hi_idx = edges_idx[i + 1]
        bins.append({
            'bin_idx':    i,
            'lo_idx':     lo_idx,
            'hi_idx':     hi_idx,
            'x_lo':       fdata['x_original'][lo_idx],
            'x_hi':       fdata['x_original'][hi_idx],
            'y_mean_avg': float(np.mean(fdata['y_mean'][lo_idx:hi_idx + 1])),
            'gradient':   float(np.mean(np.abs(np.gradient(fdata['y_mean']))[lo_idx:hi_idx + 1])),
            'n_points':   hi_idx - lo_idx + 1,
            'delta':      0.0,
            'user_n_bins': None  # Step 3: 人工覆蓋 bin 數，None = 使用自動值
        })

    fdata['adaptive_bins'] = bins
    fdata['n_adaptive_bins'] = n_actual
    return fdata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',   default='./shape_data.json')
    parser.add_argument('--output',  default='./shape_data_adaptive.json')
    parser.add_argument('--n_bins',  type=int, default=20,
                        help='Target number of bins per feature')
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    print(f"Processing {args.input}...")
    for fold_key, shapes in data['folds'].items():
        print(f"\n  Fold {fold_key}:")
        for fname, fdata in shapes.items():
            fdata = build_adaptive_feature(fdata, args.n_bins)
            n = fdata['n_adaptive_bins']
            # 找梯度最大的 bin（最需要細分的區間）
            top_bin = max(fdata['adaptive_bins'], key=lambda b: b['gradient'])
            print(f"    {fname:25s} → {n:2d} bins  "
                  f"(densest at x=[{top_bin['x_lo']:.1f},{top_bin['x_hi']:.1f}] "
                  f"grad={top_bin['gradient']:.4f})")
        data['folds'][fold_key] = shapes

    data['adaptive'] = True
    data['adaptive_n_bins'] = args.n_bins

    with open(args.output, 'w') as f:
        json.dump(data, f, indent=2)

    size_mb = __import__('os').path.getsize(args.output) / 1e6
    print(f"\n✓ Saved: {args.output}  ({size_mb:.2f} MB)")
    print("  → 把這個檔案上傳到 GitHub 的 doc/ 資料夾替換舊的 shape_data.json")


if __name__ == '__main__':
    main()
