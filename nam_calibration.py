"""
NAM-LSS Post-hoc Calibration Table
====================================
兩層校正系統，完全不需要重新訓練模型：

Layer 1 - Feature Contribution Calibration:
    對每個特徵的 Shapley-style contribution 做 piecewise-linear 調整

Layer 2 - Output Score Calibration:
    對最終 prediction 的機率區間做 bin-level 調整（類 EBM 風格）

使用方式:
    cal = NAMCalibration(model, feature_names)
    cal.fit_from_data(X_val, y_val)   # 從驗證集自動建立 table
    cal.set_feature_adjustment('Pre_HD_SBP', bin_idx=3, delta=+0.05)
    cal.set_output_adjustment(bin_idx=7, delta=-0.02)
    pred_cal = cal.predict(X_test)
    cal.save('calibration.json')
    cal.load('calibration.json')
"""

import numpy as np
import torch
import json
import os
from typing import Optional, List, Dict, Tuple
from copy import deepcopy


# ============================================================================
# Feature Contribution Extractor
# ============================================================================
class FeatureContributionExtractor:
    """
    從訓練好的 NAM 模型反推每個特徵的 contribution。
    使用 ablation 方式：contribution_i = pred(x) - pred(x with feature_i = baseline)
    這比 Shapley 快，且對 NAM 結構來說已足夠準確。
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()

    def extract(self, X: np.ndarray, baseline: Optional[np.ndarray] = None) -> np.ndarray:
        """
        X: (N, F) numpy array
        return: (N, F) contribution array，每格是該特徵對最終 prediction 的貢獻
        """
        N, F = X.shape
        x_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            full_pred = self._predict(x_t)  # (N,)

        if baseline is None:
            baseline = np.zeros(F)  # quantile-transformed 空間的 0 ≈ median
        base_t = torch.FloatTensor(baseline).to(self.device)

        contributions = np.zeros((N, F), dtype=np.float32)
        for i in range(F):
            x_ablated = x_t.clone()
            x_ablated[:, i] = base_t[i]
            with torch.no_grad():
                ablated_pred = self._predict(x_ablated)
            contributions[:, i] = (full_pred - ablated_pred).cpu().numpy()

        return contributions

    def _predict(self, x_t):
        """對 5 個 ensemble 模型取平均"""
        if hasattr(self.model, 'models'):
            preds = []
            for m in self.model.models:
                p, _, _ = m.predict_proba(x_t)
                preds.append(p)
            return torch.stack(preds).mean(0)
        else:
            p, _, _ = self.model.predict_proba(x_t)
            return p


# ============================================================================
# Layer 1: Feature-level Calibration Table
# ============================================================================
class FeatureCalibrationTable:
    """
    對每個特徵的 contribution 做 piecewise-linear 校正。
    每個特徵有 N_BINS 個 bin，每個 bin 存一個 delta 調整值。

    調整方式：
        adjusted_contribution = original_contribution + delta[bin_idx]
    """

    def __init__(self, feature_names: List[str], n_bins: int = 10):
        self.feature_names = feature_names
        self.n_bins = n_bins
        self.n_features = len(feature_names)

        # bin edges（在 contribution 空間）
        self.bin_edges: Dict[str, np.ndarray] = {}
        # delta 調整值
        self.deltas: Dict[str, np.ndarray] = {}
        # 各 bin 的統計資訊（供 UI 顯示）
        self.bin_stats: Dict[str, List[Dict]] = {}

    def fit(self, contributions: np.ndarray, y_true: np.ndarray):
        """
        從驗證集的 contribution 矩陣建立 bin 結構。
        contributions: (N, F)
        y_true: (N,)
        """
        for i, fname in enumerate(self.feature_names):
            c = contributions[:, i]
            edges = np.percentile(c, np.linspace(0, 100, self.n_bins + 1))
            # 確保 edges 唯一
            edges = np.unique(edges)
            if len(edges) < 2:
                edges = np.array([c.min() - 1e-6, c.max() + 1e-6])
            self.bin_edges[fname] = edges
            n_actual_bins = len(edges) - 1
            self.deltas[fname] = np.zeros(n_actual_bins, dtype=np.float32)

            # 計算各 bin 的統計資訊
            stats = []
            bin_ids = np.digitize(c, edges[1:-1])
            for b in range(n_actual_bins):
                mask = bin_ids == b
                stats.append({
                    'bin_idx': b,
                    'lo': float(edges[b]),
                    'hi': float(edges[b + 1]),
                    'n_samples': int(mask.sum()),
                    'mean_contribution': float(c[mask].mean()) if mask.sum() > 0 else 0.0,
                    'mean_label': float(y_true[mask].mean()) if mask.sum() > 0 else 0.0,
                    'delta': 0.0
                })
            self.bin_stats[fname] = stats

    def get_bin_idx(self, fname: str, contribution_value: float) -> int:
        edges = self.bin_edges[fname]
        return int(np.clip(np.digitize(contribution_value, edges[1:-1]), 0, len(edges) - 2))

    def adjust(self, contributions: np.ndarray) -> np.ndarray:
        """
        套用所有 delta 調整，回傳校正後的 contribution 矩陣。
        contributions: (N, F) → adjusted: (N, F)
        """
        adjusted = contributions.copy()
        for i, fname in enumerate(self.feature_names):
            if fname not in self.deltas:
                continue
            edges = self.bin_edges[fname]
            bin_ids = np.digitize(contributions[:, i], edges[1:-1])
            bin_ids = np.clip(bin_ids, 0, len(self.deltas[fname]) - 1)
            for b, delta in enumerate(self.deltas[fname]):
                mask = bin_ids == b
                adjusted[mask, i] += delta
        return adjusted

    def set_delta(self, fname: str, bin_idx: int, delta: float):
        if fname not in self.deltas:
            raise ValueError(f"Feature '{fname}' not found.")
        self.deltas[fname][bin_idx] = delta
        self.bin_stats[fname][bin_idx]['delta'] = delta

    def reset(self, fname: Optional[str] = None):
        if fname:
            self.deltas[fname][:] = 0.0
            for s in self.bin_stats[fname]:
                s['delta'] = 0.0
        else:
            for f in self.feature_names:
                if f in self.deltas:
                    self.deltas[f][:] = 0.0
                    for s in self.bin_stats[f]:
                        s['delta'] = 0.0

    def to_dict(self) -> dict:
        return {
            'feature_names': self.feature_names,
            'n_bins': self.n_bins,
            'bin_edges': {k: v.tolist() for k, v in self.bin_edges.items()},
            'deltas': {k: v.tolist() for k, v in self.deltas.items()},
            'bin_stats': self.bin_stats
        }

    @classmethod
    def from_dict(cls, d: dict):
        obj = cls(d['feature_names'], d['n_bins'])
        obj.bin_edges = {k: np.array(v) for k, v in d['bin_edges'].items()}
        obj.deltas = {k: np.array(v) for k, v in d['deltas'].items()}
        obj.bin_stats = d['bin_stats']
        return obj


# ============================================================================
# Layer 2: Output-level Calibration Table
# ============================================================================
class OutputCalibrationTable:
    """
    對最終 prediction 機率做 bin-level 調整，類似 EBM 的 score 調整。

    調整方式：
        calibrated_pred = clip(original_pred + delta[bin_idx], 0.001, 0.999)
    """

    def __init__(self, n_bins: int = 20):
        self.n_bins = n_bins
        self.bin_edges = np.linspace(0, 1, n_bins + 1)
        self.deltas = np.zeros(n_bins, dtype=np.float32)
        self.bin_stats: List[Dict] = []

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray):
        """建立 bin 統計資訊"""
        self.bin_stats = []
        bin_ids = np.digitize(y_pred, self.bin_edges[1:-1])
        for b in range(self.n_bins):
            mask = bin_ids == b
            self.bin_stats.append({
                'bin_idx': b,
                'lo': float(self.bin_edges[b]),
                'hi': float(self.bin_edges[b + 1]),
                'n_samples': int(mask.sum()),
                'mean_pred': float(y_pred[mask].mean()) if mask.sum() > 0 else float((self.bin_edges[b] + self.bin_edges[b+1]) / 2),
                'mean_label': float(y_true[mask].mean()) if mask.sum() > 0 else float((self.bin_edges[b] + self.bin_edges[b+1]) / 2),
                'calibration_gap': float(y_true[mask].mean() - y_pred[mask].mean()) if mask.sum() > 0 else 0.0,
                'delta': 0.0
            })

    def get_bin_idx(self, pred_value: float) -> int:
        return int(np.clip(np.digitize(pred_value, self.bin_edges[1:-1]), 0, self.n_bins - 1))

    def adjust(self, y_pred: np.ndarray) -> np.ndarray:
        bin_ids = np.digitize(y_pred, self.bin_edges[1:-1])
        bin_ids = np.clip(bin_ids, 0, self.n_bins - 1)
        adjusted = y_pred.copy()
        for b, delta in enumerate(self.deltas):
            mask = bin_ids == b
            adjusted[mask] += delta
        return np.clip(adjusted, 0.001, 0.999)

    def set_delta(self, bin_idx: int, delta: float):
        self.deltas[bin_idx] = delta
        self.bin_stats[bin_idx]['delta'] = delta

    def auto_calibrate(self):
        """自動根據 calibration_gap 填入建議 delta"""
        for b, stats in enumerate(self.bin_stats):
            if stats['n_samples'] > 10:
                self.deltas[b] = float(stats['calibration_gap']) * 0.5
                self.bin_stats[b]['delta'] = self.deltas[b]

    def reset(self):
        self.deltas[:] = 0.0
        for s in self.bin_stats:
            s['delta'] = 0.0

    def to_dict(self) -> dict:
        return {
            'n_bins': self.n_bins,
            'bin_edges': self.bin_edges.tolist(),
            'deltas': self.deltas.tolist(),
            'bin_stats': self.bin_stats
        }

    @classmethod
    def from_dict(cls, d: dict):
        obj = cls(d['n_bins'])
        obj.bin_edges = np.array(d['bin_edges'])
        obj.deltas = np.array(d['deltas'])
        obj.bin_stats = d['bin_stats']
        return obj


# ============================================================================
# Main Calibration Class
# ============================================================================
class NAMCalibration:
    """
    NAM-LSS Post-hoc Calibration 主介面。
    完全不動原始模型權重。

    用法:
        # 初始化
        cal = NAMCalibration(trainer, feature_names, device)

        # 從驗證集建立 table（只需跑一次）
        cal.fit_from_loader(val_loader)

        # 手動調整（特徵層）
        cal.feature_table.set_delta('Pre_HD_SBP', bin_idx=3, delta=+0.05)

        # 手動調整（輸出層）
        cal.output_table.set_delta(bin_idx=15, delta=-0.02)

        # 自動輸出校正（根據 calibration gap）
        cal.output_table.auto_calibrate()

        # 推論
        result = cal.predict(X_test)
        # result['pred_raw']       原始預測
        # result['pred_cal']       校正後預測
        # result['confidence']     信心分數
        # result['contributions']  各特徵貢獻

        # 儲存/載入
        cal.save('my_calibration.json')
        cal.load('my_calibration.json')
    """

    def __init__(self, trainer, feature_names: List[str], device=None,
                 n_feature_bins: int = 10, n_output_bins: int = 20):
        self.trainer = trainer
        self.feature_names = feature_names
        self.device = device or torch.device('cpu')
        self.extractor = FeatureContributionExtractor(trainer, self.device)
        self.feature_table = FeatureCalibrationTable(feature_names, n_feature_bins)
        self.output_table = OutputCalibrationTable(n_output_bins)
        self._is_fitted = False

    def fit_from_loader(self, val_loader):
        """從 DataLoader 建立兩層 calibration table"""
        print("🔧 Building calibration tables from validation set...")
        all_X, all_y, all_pred = [], [], []

        for xb, yb in val_loader:
            xb = xb.to(self.device)
            with torch.no_grad():
                preds = []
                for m in self.trainer.models:
                    p, _, _ = m.predict_proba(xb)
                    preds.append(p)
                pred = torch.stack(preds).mean(0).cpu().numpy()
            all_X.append(xb.cpu().numpy())
            all_y.append(yb.numpy())
            all_pred.append(pred)

        X = np.concatenate(all_X)
        y = np.concatenate(all_y)
        y_pred = np.concatenate(all_pred)

        print(f"  Extracting feature contributions for {len(self.feature_names)} features...")
        contributions = self.extractor.extract(X)

        print("  Fitting feature-level table...")
        self.feature_table.fit(contributions, y)

        print("  Fitting output-level table...")
        self.output_table.fit(y_pred, y)

        self._is_fitted = True
        print("✓ Calibration tables ready.\n")
        return self

    def fit_from_arrays(self, X_val: np.ndarray, y_val: np.ndarray):
        """從 numpy array 建立 calibration table（批次推論用）"""
        print("🔧 Building calibration tables...")
        batch_size = 1024
        preds = []
        for i in range(0, len(X_val), batch_size):
            xb = torch.FloatTensor(X_val[i:i+batch_size]).to(self.device)
            with torch.no_grad():
                ep = []
                for m in self.trainer.models:
                    p, _, _ = m.predict_proba(xb)
                    ep.append(p)
            preds.append(torch.stack(ep).mean(0).cpu().numpy())
        y_pred = np.concatenate(preds)

        contributions = self.extractor.extract(X_val)
        self.feature_table.fit(contributions, y_val)
        self.output_table.fit(y_pred, y_val)
        self._is_fitted = True
        print("✓ Done.\n")
        return self

    def predict(self, X: np.ndarray) -> Dict:
        """
        完整推論流程，回傳原始 + 校正結果。
        """
        batch_size = 1024
        all_pred, all_al, all_ep = [], [], []

        for i in range(0, len(X), batch_size):
            xb = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
            with torch.no_grad():
                ep_p, ep_al, ep_ep = [], [], []
                for m in self.trainer.models:
                    p, a, e = m.predict_proba(xb)
                    ep_p.append(p.cpu().numpy())
                    ep_al.append(a.cpu().numpy())
                    ep_ep.append(p.cpu().numpy())
            all_pred.append(np.stack(ep_p).mean(0))
            all_al.append(np.stack(ep_al).mean(0))
            all_ep.append(np.stack(ep_p).std(0))

        pred_raw = np.concatenate(all_pred)
        aleatoric = np.concatenate(all_al)
        epistemic = np.concatenate(all_ep)

        # Confidence
        total_unc = aleatoric + epistemic * 5
        if total_unc.max() > 0:
            confidence = 1.0 - total_unc / (total_unc.max() + 1e-8)
        else:
            confidence = np.ones_like(total_unc)

        # Layer 1: Feature contribution 校正
        contributions = self.extractor.extract(X)
        adj_contributions = self.feature_table.adjust(contributions)
        contrib_delta = adj_contributions.sum(axis=1) - contributions.sum(axis=1)
        pred_after_feat = np.clip(pred_raw + contrib_delta, 0.001, 0.999)

        # Layer 2: Output 校正
        pred_cal = self.output_table.adjust(pred_after_feat)

        return {
            'pred_raw': pred_raw,
            'pred_after_feature_cal': pred_after_feat,
            'pred_cal': pred_cal,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'confidence': confidence,
            'contributions_original': contributions,
            'contributions_adjusted': adj_contributions,
        }

    def summary(self):
        """印出目前所有非零調整"""
        print("\n" + "="*60)
        print("NAM Calibration Summary")
        print("="*60)
        print(f"Status: {'✓ Fitted' if self._is_fitted else '✗ Not fitted'}")
        print(f"\n[Layer 1] Feature Adjustments:")
        any_feat = False
        for fname in self.feature_names:
            if fname in self.feature_table.deltas:
                nz = np.nonzero(self.feature_table.deltas[fname])[0]
                if len(nz):
                    any_feat = True
                    for b in nz:
                        d = self.feature_table.deltas[fname][b]
                        stats = self.feature_table.bin_stats[fname][b]
                        print(f"  {fname:30s} bin {b:2d} [{stats['lo']:+.3f}, {stats['hi']:+.3f}]  delta={d:+.4f}")
        if not any_feat:
            print("  (none)")

        print(f"\n[Layer 2] Output Adjustments:")
        nz = np.nonzero(self.output_table.deltas)[0]
        if len(nz):
            for b in nz:
                d = self.output_table.deltas[b]
                s = self.output_table.bin_stats[b]
                print(f"  bin {b:2d} [{s['lo']:.2f}, {s['hi']:.2f}]  delta={d:+.4f}  "
                      f"(gap was {s['calibration_gap']:+.4f}, n={s['n_samples']})")
        else:
            print("  (none)")
        print("="*60 + "\n")

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        d = {
            'feature_names': self.feature_names,
            'feature_table': self.feature_table.to_dict(),
            'output_table': self.output_table.to_dict(),
            'is_fitted': self._is_fitted,
            'version': 'nam_cal_v1'
        }
        with open(path, 'w') as f:
            json.dump(d, f, indent=2)
        print(f"✓ Calibration saved to {path}")

    def load(self, path: str):
        with open(path) as f:
            d = json.load(f)
        self.feature_table = FeatureCalibrationTable.from_dict(d['feature_table'])
        self.output_table = OutputCalibrationTable.from_dict(d['output_table'])
        self._is_fitted = d.get('is_fitted', True)
        print(f"✓ Calibration loaded from {path}")
        return self

    def export_for_ui(self) -> dict:
        """匯出給 Web UI 使用的完整資料結構"""
        return {
            'feature_table': self.feature_table.to_dict(),
            'output_table': self.output_table.to_dict(),
        }
