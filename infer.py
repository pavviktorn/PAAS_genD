#!/usr/bin/env python3.12
"""GenD inference.

Single image:
    python3.12 infer.py --ckpt runs/gend_default/best.pt --image /path/face.jpg
Full eval on a MIDS json (acc/bin_auc/ap/ovr + operating point):
    python3.12 infer.py --ckpt runs/gend_default/best.pt --eval
    python3.12 infer.py --ckpt runs/gend_default/best.pt --val other.json

The checkpoint stores only the trainable subset (LayerNorm + head); the frozen
CLIP backbone is rebuilt from `clip_base` and the finetuned LN/head are loaded on
top. Threshold defaults to the checkpoint's threshold@real95 (else 0.5).
"""

import argparse
import json
import os

import torch

from gend.data import MidsDataset, build_index, build_transforms
from gend.engine import evaluate, fake_probability
from gend.metrics import format_metrics
from gend.model import GenDModel
from gend.utils import amp_dtype_from_str, set_cpu_threads


def load_model(ckpt_path, device, amp_dtype):
    ck = torch.load(ckpt_path, map_location="cpu")
    mc = ck["config"]
    model = GenDModel(ck["clip_base"], num_classes=mc["num_classes"],
                      head=ck.get("head", "LinearNorm"),
                      unfreeze_layers=mc.get("unfreeze_layers")).to(device)
    model.load_trainable(ck)
    model.eval()
    return model, ck, mc


def resolve_threshold(args, ck):
    if args.threshold is not None:
        return args.threshold
    for k, v in (ck.get("metrics", {}) or {}).items():
        if k.startswith("threshold@real"):
            return float(v)
    return 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--image", default=None)
    ap.add_argument("--val", default=None)
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    set_cpu_threads(cfg.get("omp_threads", 8))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = amp_dtype_from_str(cfg.get("amp_dtype", "bf16"))
    model, ck, mc = load_model(args.ckpt, device, amp_dtype)
    thr = resolve_threshold(args, ck)
    num_classes = mc["num_classes"]
    image_size = cfg.get("image_size", 224)

    if args.image:
        from gend.data import class_names
        from PIL import Image
        tfm = build_transforms(image_size, train=False)
        x = tfm(Image.open(args.image).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=amp_dtype,
                                             enabled=(device.type == "cuda" and amp_dtype != torch.float32)):
            out = model(x)
        probs = torch.softmax(out.logits.float(), dim=1)[0]
        p_fake = float(fake_probability(probs.unsqueeze(0), num_classes)[0])
        names = class_names(num_classes)
        print(json.dumps({
            "image": args.image, "p_fake": round(p_fake, 6), "threshold": round(thr, 6),
            "verdict": "FAKE" if p_fake >= thr else "REAL",
            "type": names[int(probs.argmax())],
            "probs": {n: round(float(v), 6) for n, v in zip(names, probs.tolist())},
        }, indent=2))
        return

    val_json = args.val or (cfg["val_data"] if args.eval else None)
    if not val_json:
        ap.error("provide --image, or --val <json>, or --eval")
    samples = build_index(val_json, num_classes)
    ds = MidsDataset(samples, build_transforms(image_size, train=False), image_size, return_index=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                         num_workers=args.num_workers, pin_memory=True)
    m = evaluate(model, loader, device, amp_dtype, num_classes, is_dist=False, world_size=1,
                 real_recall_target=cfg.get("real_recall_target", 0.95))
    print(f"checkpoint: {args.ckpt}  (step {ck.get('step')}, epoch {ck.get('epoch')})")
    print(f"eval on: {val_json}")
    print(format_metrics(m))


if __name__ == "__main__":
    main()
