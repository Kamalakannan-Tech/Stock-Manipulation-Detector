"""
Evaluation module — F1, ROC-AUC, confusion matrix, precision-recall, plots.
"""
import numpy as np
import pandas as pd
import logging
import json
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    from sklearn.metrics import (
        f1_score, roc_auc_score, roc_curve, precision_score,
        recall_score, confusion_matrix, precision_recall_curve,
        average_precision_score,
    )
    HAS_SKL = True
except ImportError:
    HAS_SKL = False


PLOTS_DIR = Path('data/evaluation')


class ManipulationEvaluator:
    """Full evaluation suite for pump-and-dump detection."""

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def compute_metrics(self, y_true: np.ndarray, y_prob: np.ndarray) -> dict:
        """Compute all classification metrics."""
        y_pred = (y_prob >= self.threshold).astype(int)

        metrics = {
            'threshold':     self.threshold,
            'n_samples':     int(len(y_true)),
            'n_positive':    int(y_true.sum()),
            'positive_rate': float(y_true.mean()),
            'timestamp':     datetime.now().isoformat(),
        }

        if not HAS_SKL:
            logger.warning('scikit-learn not available; computing basic metrics only')
            tp = int(((y_pred == 1) & (y_true == 1)).sum())
            fp = int(((y_pred == 1) & (y_true == 0)).sum())
            fn = int(((y_pred == 0) & (y_true == 1)).sum())
            tn = int(((y_pred == 0) & (y_true == 0)).sum())
            prec   = tp / (tp + fp + 1e-9)
            rec    = tp / (tp + fn + 1e-9)
            f1     = 2 * prec * rec / (prec + rec + 1e-9)
            acc    = (tp + tn) / len(y_true)
            fpr    = fp / (fp + tn + 1e-9)
            metrics.update({'f1': f1, 'precision': prec, 'recall': rec,
                            'accuracy': acc, 'false_alarm_rate': fpr,
                            'confusion_matrix': [[tn, fp], [fn, tp]]})
            return metrics

        try:
            f1        = f1_score(y_true, y_pred, zero_division=0)
            prec      = precision_score(y_true, y_pred, zero_division=0)
            rec       = recall_score(y_true, y_pred, zero_division=0)
            auc       = roc_auc_score(y_true, y_prob) if y_true.sum() > 0 else 0.0
            avg_prec  = average_precision_score(y_true, y_prob) if y_true.sum() > 0 else 0.0
            cm        = confusion_matrix(y_true, y_pred).tolist()
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
            fpr       = fp / (fp + tn + 1e-9)
            acc       = (tp + tn) / len(y_true)
            metrics.update({
                'f1':               round(float(f1),       4),
                'precision':        round(float(prec),     4),
                'recall':           round(float(rec),      4),
                'roc_auc':          round(float(auc),      4),
                'avg_precision':    round(float(avg_prec), 4),
                'accuracy':         round(float(acc),      4),
                'false_alarm_rate': round(float(fpr),      4),
                'confusion_matrix': cm,
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            })
        except Exception as e:
            logger.error(f'Metrics computation error: {e}')

        return metrics

    # ------------------------------------------------------------------
    def find_best_threshold(self, y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """Find threshold that maximises a blend of F1 and Accuracy."""
        if not HAS_SKL:
            return 0.5
        thresholds = np.linspace(0.1, 0.9, 33)
        best_t, best_score, best_f1 = 0.5, 0.0, 0.0
        for t in thresholds:
            pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, pred, zero_division=0)
            acc = (pred == y_true).mean()
            score = (0.7 * f1) + (0.3 * acc)
            if score > best_score:
                best_score, best_t, best_f1 = score, t, f1
        logger.info(f'Best threshold: {best_t:.2f}  (F1={best_f1:.4f}, Blended Score={best_score:.4f})')
        return float(best_t)

    # ------------------------------------------------------------------
    def save_metrics(self, metrics: dict, path: str = 'data/evaluation/metrics.json'):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f'Metrics saved → {path}')

    # ------------------------------------------------------------------
    def plot_all(self, y_true: np.ndarray, y_prob: np.ndarray,
                 training_history: dict | None = None) -> dict:
        """Generate confusion matrix, ROC curve, PR curve, training loss plots."""
        if not HAS_MPL:
            logger.warning('matplotlib not available; skipping plots')
            return {}

        plot_paths = {}

        # 1. Confusion matrix
        p = self._plot_confusion_matrix(y_true, y_prob)
        if p: plot_paths['confusion_matrix'] = p

        # 2. ROC curve
        p = self._plot_roc_curve(y_true, y_prob)
        if p: plot_paths['roc_curve'] = p

        # 3. Precision-Recall curve
        p = self._plot_pr_curve(y_true, y_prob)
        if p: plot_paths['pr_curve'] = p

        # 4. Training history
        if training_history:
            p = self._plot_training(training_history)
            if p: plot_paths['training'] = p

        # 5. Score distribution
        p = self._plot_score_dist(y_true, y_prob)
        if p: plot_paths['score_dist'] = p

        return plot_paths

    # ------------------------------------------------------------------
    def _plot_confusion_matrix(self, y_true, y_prob) -> str | None:
        try:
            y_pred = (y_prob >= self.threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.colorbar(im, ax=ax)
            ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
            ax.set_xticklabels(['Normal', 'Manipulation'])
            ax.set_yticklabels(['Normal', 'Manipulation'])
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('Actual', fontsize=12)
            ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                            fontsize=16, color='white' if cm[i, j] > cm.max()/2 else 'black')
            plt.tight_layout()
            path = str(PLOTS_DIR / 'confusion_matrix.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return path
        except Exception as e:
            logger.error(f'Confusion matrix plot error: {e}')
            return None

    def _plot_roc_curve(self, y_true, y_prob) -> str | None:
        try:
            if y_true.sum() == 0: return None
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            auc_val = roc_auc_score(y_true, y_prob)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, lw=2, label=f'AUC = {auc_val:.3f}', color='#2196F3')
            ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
            ax.fill_between(fpr, tpr, alpha=0.15, color='#2196F3')
            ax.set_xlabel('False Positive Rate', fontsize=12)
            ax.set_ylabel('True Positive Rate', fontsize=12)
            ax.set_title('ROC Curve', fontsize=14, fontweight='bold')
            ax.legend(loc='lower right', fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            path = str(PLOTS_DIR / 'roc_curve.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return path
        except Exception as e:
            logger.error(f'ROC plot error: {e}')
            return None

    def _plot_pr_curve(self, y_true, y_prob) -> str | None:
        try:
            if y_true.sum() == 0: return None
            prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(rec_arr, prec_arr, lw=2, color='#4CAF50', label=f'AP = {ap:.3f}')
            ax.fill_between(rec_arr, prec_arr, alpha=0.15, color='#4CAF50')
            ax.set_xlabel('Recall', fontsize=12)
            ax.set_ylabel('Precision', fontsize=12)
            ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            path = str(PLOTS_DIR / 'pr_curve.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return path
        except Exception as e:
            logger.error(f'PR curve plot error: {e}')
            return None

    def _plot_training(self, history: dict) -> str | None:
        try:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            if 'train_loss' in history:
                axes[0].plot(history['train_loss'], label='Train', color='#2196F3')
            if 'val_loss' in history:
                axes[0].plot(history['val_loss'], label='Val', color='#FF5722')
            axes[0].set_title('Loss', fontweight='bold')
            axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
            if 'val_f1' in history:
                axes[1].plot(history['val_f1'], label=f"Val F1 (best={max(history['val_f1']):.3f})", color='#4CAF50')
            if 'val_auc' in history:
                axes[1].plot(history['val_auc'], label=f"Val AUC (best={max(history['val_auc']):.3f})", color='#9C27B0')
            if 'val_precision' in history:
                axes[1].plot(history['val_precision'], label='Val Prec', color='#FF9800', alpha=0.7, linestyle='--')
            if 'val_recall' in history:
                axes[1].plot(history['val_recall'], label='Val Rec', color='#03A9F4', alpha=0.7, linestyle='--')
            axes[1].set_title('Metrics', fontweight='bold')
            axes[1].set_ylim(0, 1.05)
            axes[1].set_xlabel('Epoch'); axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
            plt.tight_layout()
            path = str(PLOTS_DIR / 'training_history.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return path
        except Exception as e:
            logger.error(f'Training history plot error: {e}')
            return None

    def _plot_score_dist(self, y_true, y_prob) -> str | None:
        try:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(y_prob[y_true == 0], bins=40, alpha=0.6, color='#2196F3', label='Normal')
            ax.hist(y_prob[y_true == 1], bins=40, alpha=0.6, color='#F44336', label='Manipulation')
            ax.axvline(self.threshold, color='black', linestyle='--', label=f'Threshold={self.threshold}')
            ax.set_xlabel('Manipulation Probability', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title('Score Distribution', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            path = str(PLOTS_DIR / 'score_distribution.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            return path
        except Exception as e:
            logger.error(f'Score dist plot error: {e}')
            return None
