"""Shared distributed evaluation for GenD."""

import numpy as np
import torch

from .data import class_names
from .get_label import REAL
from .metrics import compute_metrics
from .utils import all_gather_object


@torch.no_grad()
def fake_probability(probs, num_classes):
    if num_classes == 2:
        return probs[:, 1]
    return 1.0 - probs[:, REAL]


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype, num_classes,
             is_dist=False, world_size=1, real_recall_target=0.95):
    model.eval()
    local = {}  # idx -> (fake_score, true_mc, pred_mc, probs_list)
    for x, label, idx in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
            out = model(x)
        probs = torch.softmax(out.logits.float(), dim=1)
        fscore = fake_probability(probs, num_classes).cpu().numpy()
        pred = probs.argmax(dim=1).cpu().numpy()
        probs = probs.cpu().numpy()
        label = label.numpy(); idx = idx.numpy()
        for j in range(len(idx)):
            local[int(idx[j])] = (float(fscore[j]), int(label[j]), int(pred[j]), probs[j].tolist())

    merged = {}
    for d in all_gather_object(local, is_dist, world_size):
        merged.update(d)
    ks = list(merged.keys())
    s = np.array([merged[k][0] for k in ks], dtype=np.float64)
    true_mc = np.array([merged[k][1] for k in ks], dtype=np.int64)
    pred_mc = np.array([merged[k][2] for k in ks], dtype=np.int64)
    probs = np.array([merged[k][3] for k in ks], dtype=np.float64)
    bin_labels = (true_mc != REAL).astype(np.int64)
    return compute_metrics(s, bin_labels, probs=probs, true_mc=true_mc, pred_mc=pred_mc,
                           class_names=class_names(num_classes),
                           real_recall_target=real_recall_target)
