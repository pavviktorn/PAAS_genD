#!/usr/bin/env python3.12
"""Train GenD on MIDS (frozen CLIP ViT-L/14, LayerNorm-only + L2-normalized head,
CE + uniformity + alignment). Multi-GPU DDP via torch.distributed.run.

Init: the released GenD checkpoint (yermandy/GenD_CLIP_L_14) provides the encoder
(finetuned LayerNorm baked into a full CLIP vision tower); a fresh num_classes
head is trained. Labels are path-derived via get_label_all (3-class real/pad/deepfake).
"""

import argparse
import collections
import json
import math
import os
import random
import time

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from gend.data import MidsDataset, build_index, build_transforms
from gend.engine import evaluate
from gend.losses import GenDLoss
from gend.metrics import format_metrics
from gend.model import GenDModel
from gend.utils import (amp_dtype_from_str, barrier, init_distributed, is_main,
                        seed_everything, set_cpu_threads)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.json")
    for k in ["epochs", "batch_size", "num_workers", "eval_every_steps",
              "warmup_steps", "num_classes"]:
        ap.add_argument(f"--{k}", type=int, default=None)
    for k in ["lr", "min_lr", "weight_decay", "real_recall_target"]:
        ap.add_argument(f"--{k}", type=float, default=None)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--amp_dtype", default=None)
    ap.add_argument("--select_metric", default=None)
    ap.add_argument("--init_from", default=None, help="HF id or local .safetensors for encoder init; '' to skip")
    ap.add_argument("--limit_train", type=int, default=None)
    ap.add_argument("--limit_val", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    return ap.parse_args()


def load_config(args):
    with open(args.config) as f:
        cfg = json.load(f)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    for k, v in vars(args).items():
        if k == "config" or v is None:
            continue
        cfg[k] = v
    return cfg


def resolve_init_path(init_from, log):
    if not init_from:
        return None
    if os.path.isfile(init_from):
        return init_from
    from huggingface_hub import hf_hub_download
    log(f"[init] fetching {init_from}/model.safetensors ...")
    return hf_hub_download(init_from, "model.safetensors")


def make_loader(samples, cfg, train, rank, world_size, is_dist):
    tfm = build_transforms(cfg["image_size"], train=train)
    ds = MidsDataset(samples, tfm, cfg["image_size"], return_index=not train)
    sampler = DistributedSampler(ds, world_size, rank, shuffle=train, drop_last=train) if is_dist else None
    loader = DataLoader(ds, batch_size=cfg["batch_size"], sampler=sampler,
                        shuffle=(sampler is None and train), drop_last=train,
                        num_workers=cfg["num_workers"], pin_memory=True,
                        persistent_workers=cfg["num_workers"] > 0,
                        prefetch_factor=4 if cfg["num_workers"] > 0 else None)
    return sampler, loader


def main():
    args = parse_args()
    cfg = load_config(args)
    rank, world_size, local_rank, device, is_dist = init_distributed()
    set_cpu_threads(cfg["omp_threads"])
    seed_everything(cfg["seed"], rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    amp_dtype = amp_dtype_from_str(cfg["amp_dtype"])

    def log(*a):
        if is_main(rank):
            print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)

    if is_main(rank):
        os.makedirs(cfg["out_dir"], exist_ok=True)
        json.dump(cfg, open(os.path.join(cfg["out_dir"], "config.used.json"), "w"), indent=2)
        log("config:", json.dumps(cfg, indent=2))
        log(f"world_size={world_size} device={device} amp={cfg['amp_dtype']}")

    # indexes (rank0 builds cache, others read)
    if is_main(rank):
        train_samples = build_index(cfg["train_data"], cfg["num_classes"], log=log)
        val_samples = build_index(cfg["val_data"], cfg["num_classes"], log=log)
    barrier(is_dist)
    if not is_main(rank):
        train_samples = build_index(cfg["train_data"], cfg["num_classes"], log=lambda *a: None)
        val_samples = build_index(cfg["val_data"], cfg["num_classes"], log=lambda *a: None)
    if args.limit_train:
        train_samples = random.Random(cfg["seed"]).sample(train_samples, min(args.limit_train, len(train_samples)))
    if args.limit_val:
        val_samples = random.Random(cfg["seed"]).sample(val_samples, min(args.limit_val, len(val_samples)))
    counts = collections.Counter(l for _, l in train_samples)
    log(f"train={len(train_samples)} val={len(val_samples)} counts={dict(sorted(counts.items()))}")

    train_sampler, train_loader = make_loader(train_samples, cfg, True, rank, world_size, is_dist)
    _, val_loader = make_loader(val_samples, cfg, False, rank, world_size, is_dist)

    # model (+ pretrained encoder init: rank0 downloads to cache first)
    model = GenDModel(cfg["clip_base"], num_classes=cfg["num_classes"],
                      head=cfg["head"], unfreeze_layers=cfg.get("unfreeze_layers")).to(device)
    init_from = cfg.get("init_from")
    if init_from:
        if is_main(rank):
            path = resolve_init_path(init_from, log)
        barrier(is_dist)
        if not is_main(rank):
            path = resolve_init_path(init_from, lambda *a: None)
        n, missing, unexpected = model.load_pretrained_encoder(path)
        log(f"[init] loaded {n} encoder tensors from {init_from} "
            f"(missing={len(missing)} unexpected={len(unexpected)})")
    n_train = sum(p.numel() for p in model.trainable_parameters())
    n_all = sum(p.numel() for p in model.parameters())
    log(f"trainable params: {n_train/1e6:.4f}M / {n_all/1e6:.1f}M ({100*n_train/n_all:.3f}%) "
        f"(LN groups {cfg.get('unfreeze_layers')} + head)")

    raw_model = model
    if is_dist:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    weight = None
    if cfg.get("class_weight", True):
        tot = sum(counts.values())
        w = [tot / (cfg["num_classes"] * counts.get(c, 1)) for c in range(cfg["num_classes"])]
        weight = torch.tensor(w, dtype=torch.float32, device=device)
        log(f"class weights: {[round(x,3) for x in w]}")
    crit = GenDLoss(ce_labels=cfg["loss_ce"], uniformity_w=cfg["loss_uniformity"],
                    alignment_w=cfg["loss_alignment"], label_smoothing=cfg["label_smoothing"],
                    class_weight=weight)

    opt = torch.optim.AdamW(raw_model.trainable_parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"], betas=tuple(cfg["betas"]))

    steps_per_epoch = len(train_loader)
    total_steps = args.max_steps or steps_per_epoch * cfg["epochs"]
    warmup = cfg["warmup_steps"]
    min_lr = cfg["min_lr"]
    log(f"cosine LR: peak {cfg['lr']:.2e} -> min {min_lr:.2e}, warmup {warmup}, total {total_steps}")

    def lr_at(step):
        if warmup > 0 and step < warmup:
            return min_lr + (cfg["lr"] - min_lr) * (step + 1) / warmup
        prog = min(max((step - warmup) / max(1, total_steps - warmup), 0.0), 1.0)
        return min_lr + 0.5 * (cfg["lr"] - min_lr) * (1.0 + math.cos(math.pi * prog))

    best, sel = -1.0, cfg["select_metric"]
    gstep, last_eval = 0, -1
    t0 = time.time(); stop = False
    for epoch in range(cfg["epochs"]):
        if is_dist:
            train_sampler.set_epoch(epoch)
        model.train()
        for x, y in train_loader:
            for g in opt.param_groups:
                g["lr"] = lr_at(gstep)
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=(amp_dtype != torch.float32)):
                out = model(x)
                loss = crit(out.logits, y, out.l2_embeddings)
            opt.zero_grad(set_to_none=True)
            loss["total"].backward()
            if cfg["grad_clip"] > 0:
                nn.utils.clip_grad_norm_(raw_model.trainable_parameters(), cfg["grad_clip"])
            opt.step()
            gstep += 1
            if gstep % cfg["log_every_steps"] == 0:
                rate = gstep * cfg["batch_size"] * world_size / (time.time() - t0)
                log(f"ep{epoch} step{gstep}/{total_steps} loss {loss['total'].item():.4f} "
                    f"(ce {loss.get('ce',0):.3f} uni {loss.get('uniformity',0):.3f} "
                    f"ali {loss.get('alignment',0):.3f}) lr {opt.param_groups[0]['lr']:.2e} {rate:.0f} img/s")
            if cfg["eval_every_steps"] and gstep % cfg["eval_every_steps"] == 0:
                best = _eval_save(raw_model, val_loader, device, amp_dtype, cfg, is_dist,
                                  world_size, rank, sel, best, gstep, epoch, log)
                last_eval = gstep; model.train()
            if args.max_steps and gstep >= args.max_steps:
                stop = True; break
        if stop:
            break
        if gstep != last_eval:
            best = _eval_save(raw_model, val_loader, device, amp_dtype, cfg, is_dist,
                              world_size, rank, sel, best, gstep, epoch, log)
            last_eval = gstep
    log(f"done. best {sel}={best:.4f}. checkpoints in {cfg['out_dir']}")
    if is_dist:
        torch.distributed.destroy_process_group()


def _eval_save(raw_model, val_loader, device, amp_dtype, cfg, is_dist, world_size,
               rank, sel, best, gstep, epoch, log):
    m = evaluate(raw_model, val_loader, device, amp_dtype, cfg["num_classes"],
                 is_dist=is_dist, world_size=world_size, real_recall_target=cfg["real_recall_target"])
    if is_main(rank):
        log(f"[eval] step {gstep} (ep{epoch})\n" + format_metrics(m))
        torch.save({"step": gstep, "epoch": epoch, "metrics": m, "clip_base": cfg["clip_base"],
                    "head": cfg["head"], **raw_model.export_state()},
                   os.path.join(cfg["out_dir"], "last.pt"))
        cur = m.get(sel, m.get("bin_auc"))
        if cur is not None and cur > best:
            best = cur
            torch.save({"step": gstep, "epoch": epoch, "metrics": m, "clip_base": cfg["clip_base"],
                        "head": cfg["head"], **raw_model.export_state()},
                       os.path.join(cfg["out_dir"], "best.pt"))
            log(f"[eval] new best {sel}={best:.4f} -> saved best.pt")
        with open(os.path.join(cfg["out_dir"], "metrics_log.jsonl"), "a") as f:
            f.write(json.dumps({"step": gstep, "epoch": epoch, **m}) + "\n")
    if is_dist:
        t = torch.tensor([best], device=device)
        torch.distributed.broadcast(t, src=0)
        best = float(t.item())
    return best


if __name__ == "__main__":
    main()
