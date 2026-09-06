"""Metrics for GenD. Primary = bin_auc (binary real-vs-fake AUROC, fake=positive).

Also reports: ovr_auroc (multi-class one-vs-rest macro AUROC, GenD's src/metrics.py
style), ap (binary average precision), acc, eer, per-class recall, and the
fake-recall @ real-recall floor operating points.
"""

import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score, roc_curve)


def compute_metrics(fake_scores, bin_labels, probs=None, true_mc=None, pred_mc=None,
                    class_names=None, real_recall_target=0.95):
    s = np.asarray(fake_scores, dtype=np.float64)
    y = np.asarray(bin_labels, dtype=np.int64)
    out = {"n": int(len(y)), "n_real": int((y == 0).sum()), "n_fake": int((y == 1).sum())}

    pred = (s >= 0.5).astype(int)
    out["acc"] = float((pred == y).mean())
    both = (y == 0).any() and (y == 1).any()
    if both:
        out["bin_auc"] = float(roc_auc_score(y, s))
        out["ap"] = float(average_precision_score(y, s))
        fpr, tpr, thr = roc_curve(y, s)
        fnr = 1.0 - tpr
        i = int(np.nanargmin(np.abs(fnr - fpr)))
        out["eer"] = float((fpr[i] + fnr[i]) / 2.0)
        out["eer_threshold"] = float(thr[i])

    # multi-class one-vs-rest macro AUROC (matches GenD ovr_roc)
    if probs is not None and true_mc is not None:
        P = np.asarray(probs, dtype=np.float64)
        t = np.asarray(true_mc, dtype=np.int64)
        classes = sorted(set(t.tolist()))
        if len(classes) == P.shape[1] and len(classes) > 1:
            oh = np.eye(P.shape[1])[t]
            try:
                out["ovr_auroc"] = float(roc_auc_score(oh, P, multi_class="ovr", average="macro"))
            except Exception:
                pass

    if true_mc is not None and pred_mc is not None:
        t = np.asarray(true_mc, dtype=np.int64)
        p = np.asarray(pred_mc, dtype=np.int64)
        names = class_names or [str(c) for c in sorted(set(t.tolist()))]
        for c, name in enumerate(names):
            m = t == c
            if m.any():
                out[f"recall_{name}"] = float((p[m] == c).mean())

    real, fake = s[y == 0], s[y == 1]
    if real.size and fake.size:
        for floor in (real_recall_target,):
            tag = int(round(floor * 100))
            cands = np.unique(np.concatenate([s, [s.max() + 1e-6]]))
            best = None
            for thr_ in cands:
                rr = float((real < thr_).mean())
                if rr >= floor:
                    fr = float((fake >= thr_).mean())
                    if best is None or fr > best[1]:
                        best = (float(thr_), fr, rr)
            if best:
                out[f"fake_recall@real{tag}"] = best[1]
                out[f"threshold@real{tag}"] = best[0]
                out[f"real_recall@real{tag}"] = best[2]
    return out


def fake_recall_at_floors(fake_scores, bin_labels, floors=(0.80, 0.85, 0.90, 0.95, 0.98)):
    s = np.asarray(fake_scores, dtype=np.float64)
    y = np.asarray(bin_labels, dtype=np.int64)
    real, fake = s[y == 0], s[y == 1]
    rows = []
    for f in floors:
        tau = float(np.quantile(real, f))
        rows.append((f, tau, float((real < tau).mean()), float((fake >= tau).mean())))
    return rows


def format_metrics(m):
    keys = ["n", "n_real", "n_fake", "acc", "bin_auc", "ovr_auroc", "ap", "eer",
            "eer_threshold"]
    pc = [k for k in m if k.startswith("recall_")]
    op = [k for k in m if k.startswith(("fake_recall@real", "threshold@real", "real_recall@real"))]
    lines = []
    for k in keys + pc + op:
        if k in m:
            v = m[k]
            lines.append(f"  {k:22s}: {v:.4f}" if isinstance(v, float) else f"  {k:22s}: {v}")
    return "\n".join(lines)
