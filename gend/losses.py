"""GenD loss = CE + alignment + uniformity (faithful port of src/loss.py + unifalign.py).

    L = ce_labels * CE(logits, y, label_smoothing)
      + alignment_labels * alignment(z, y)        # pull same-class L2-embeddings together
      + uniformity      * uniformity(z)           # spread all L2-embeddings on the sphere

z are unit-normalized embeddings. Defaults from the best GenD config (wacv-LN+L2+UnAl):
ce_labels=1.0, uniformity=0.5, alignment_labels=0.1.
"""

import torch
import torch.nn.functional as F


def alignment(embeddings, labels, alpha: float = 2.0):
    """Label-aware alignment: mean ||z_x - z_y||^alpha over same-label pairs (excl. self)."""
    n = embeddings.size(0)
    if n < 2:
        return embeddings.new_tensor(0.0)
    mask = (labels[:, None] == labels[None, :]).triu(diagonal=1)
    idx = torch.nonzero(mask, as_tuple=False)
    if idx.numel() == 0:
        return embeddings.new_tensor(0.0)
    x = embeddings[idx[:, 0]]
    y = embeddings[idx[:, 1]]
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniformity(x, t: float = 2.0, clip_value: float = 1e-6):
    """log E[ exp(-t * ||z_i - z_j||^2) ] over all pairs."""
    if x.size(0) < 2:
        return x.new_tensor(0.0)
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().clamp(min=clip_value).log()


class GenDLoss(torch.nn.Module):
    def __init__(self, ce_labels=1.0, uniformity_w=0.5, alignment_w=0.1,
                 label_smoothing=0.0, class_weight=None):
        super().__init__()
        self.ce_w = ce_labels
        self.uni_w = uniformity_w
        self.ali_w = alignment_w
        self.label_smoothing = label_smoothing
        self.register_buffer("class_weight", class_weight if class_weight is not None else None)

    def forward(self, logits, labels, l2_embeddings):
        out = {}
        total = logits.sum() * 0.0
        if self.ce_w:
            ce = F.cross_entropy(logits, labels, weight=self.class_weight,
                                 label_smoothing=self.label_smoothing)
            out["ce"] = ce.item()
            total = total + self.ce_w * ce
        if self.ali_w:
            al = alignment(l2_embeddings, labels)
            out["alignment"] = al.item()
            total = total + self.ali_w * al
        if self.uni_w:
            un = uniformity(l2_embeddings)
            out["uniformity"] = un.item()
            total = total + self.uni_w * un
        if torch.is_tensor(total) and total.isnan():
            total = logits.sum() * 0.0
        out["total"] = total
        return out
