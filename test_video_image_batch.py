#!/usr/bin/env python3.12
"""PAAS_genD batch tester over a folder tree of images + videos -- multi-GPU batching.

Mirrors PAAS_ensemble_v2/test_video_image_batch.py (same discovery, file-sharding
across GPUs, real cross-file batching, and the SAME unified line format) driving a
single GenD detector:

  * recurses --input-dir, finds every still image + video, FILE-SHARDS them round-robin
    across the selected GPUs (one spawn worker per GPU, pinned to its device == cuda:0);
  * REAL batching across files per --flush-size; full-image input (no face crop);
  * ground truth = the `real`/`fake` path component;
  * pred = fake if P(fake) >= threshold else real; type = argmax 3-class name; P(fake)=1-P(real);
  * merges shard results -> results_gend.txt in the unified format
    (OK/XX/SK/ER  truth=..  pred=..  type=..  fake=..  match=..  <path>) + per-label summary.

Examples:
  python3.12 test_video_image_batch.py --ckpt runs/gend_default/best.pt \
      --input-dir /datasets/work/vLLM/temp/testset --out-dir runs/test --devices 0,1,2,3
"""
import argparse
import multiprocessing as mp
import os
import sys
import time

PROJECT = "PAAS_genD"
_HERE = os.path.dirname(os.path.abspath(__file__))

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
VID_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".mpeg", ".mpg")


def fmt_line(tag, truth, pred, ftype, fake, match, path):
    fs = "------" if fake is None else f"{float(fake):.4f}"
    ms = "------" if match is None else f"{float(match):.4f}"
    tf = f"type={ftype}"
    tf = tf + " " * max(2, 15 - len(tf))
    return f"{tag}  truth={truth}  pred={pred}  {tf}fake={fs} match={ms}  {path}"


def truth_of(path):
    parts = path.lower().split(os.sep)
    if "real" in parts:
        return "real"
    if "fake" in parts:
        return "fake"
    return None


def iter_media(root, skip_dirs=()):
    skip_dirs = tuple(os.path.abspath(d) for d in skip_dirs if d)
    for dp, _, files in os.walk(root):
        adp = os.path.abspath(dp)
        if any(adp == s or adp.startswith(s + os.sep) for s in skip_dirs):
            continue
        for fn in sorted(files):
            ext = os.path.splitext(fn)[1].lower()
            kind = "image" if ext in IMG_EXT else ("video" if ext in VID_EXT else None)
            if kind is not None:
                yield os.path.join(dp, fn), kind


def frames_of(path, kind, stride):
    import cv2
    if kind == "image":
        im = cv2.imread(path, cv2.IMREAD_COLOR)
        if im is not None:
            yield path, cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return
    cap = cv2.VideoCapture(path)
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i % stride == 0:
            yield f"{path}#frame={i:06d}", cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        i += 1
    cap.release()


def miss_target(miss_dir, truth, key):
    flat = key.replace("#frame=", "_frame").lstrip("/").replace("/", "__")
    return os.path.join(miss_dir, truth, flat)


def parse_devices(args):
    import torch
    n = torch.cuda.device_count()
    if args.device is not None:
        return [int(str(args.device).strip().lower().replace("cuda:", ""))], n
    spec = args.devices.strip().lower()
    if spec == "all":
        return list(range(n)), n
    return [int(x.replace("cuda:", "")) for x in spec.split(",") if x.strip()], n


def resolve_threshold(args, ck):
    if args.threshold is not None:
        return args.threshold
    for k, v in (ck.get("metrics", {}) or {}).items():
        if k.startswith("threshold@real"):
            return float(v)
    return 0.5


def gpu_worker(device_id, media_paths, args_dict, n_devices, result_q):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    args = argparse.Namespace(**args_dict)

    import cv2
    import torch
    from PIL import Image
    sys.path.insert(0, _HERE)
    from gend.data import build_transforms, class_names
    from gend.engine import fake_probability
    from gend.utils import amp_dtype_from_str
    from infer import load_model
    import json as _json

    try:
        torch.set_num_threads(max(2, int((os.cpu_count() or 8) * 0.5) // max(1, n_devices)))
    except Exception:
        pass
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)

    cfg = _json.load(open(args.config))
    amp = amp_dtype_from_str(cfg.get("amp_dtype", "bf16"))
    device = torch.device("cuda:0")
    model, ck, mc = load_model(args.ckpt, device, amp)
    num_classes = mc["num_classes"]
    names = class_names(num_classes)
    image_size = cfg.get("image_size", 224)
    tfm = build_transforms(image_size, train=False)
    thr = resolve_threshold(args, ck)
    print(f"[GPU {device_id}] GenD ready (num_classes={num_classes}) threshold={thr:.4f} "
          f"({len(media_paths)} media files in shard)", flush=True)

    sfx = f".shard{device_id}"
    f_out = open(os.path.join(args.out_dir, f"results_gend{sfx}.txt"), "w")
    tally = {}
    counts = {"frames": 0, "skipped": 0, "errors": 0, "miss_saved": 0}
    buffer = []
    t0 = time.time(); last_report = [time.time()]

    def handle(it, p_fake, pred_cls):
        key, truth, rgb = it["key"], it["truth"], it["rgb"]
        pred = "fake" if p_fake >= thr else "real"
        correct = (pred == truth)
        f_out.write(fmt_line("OK" if correct else "XX", truth, pred, names[pred_cls], p_fake, None, key) + "\n")
        t = tally.setdefault(truth, [0, 0]); t[0] += correct; t[1] += 1
        if args.copy_miss and not correct:
            dst = miss_target(args.miss_dir, truth, key)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            try:
                cv2.imwrite(dst, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)); counts["miss_saved"] += 1
            except Exception:
                pass

    def flush():
        if not buffer:
            return
        try:
            x = torch.stack([tfm(Image.fromarray(it["rgb"])) for it in buffer]).to(device)
            with torch.autocast(device_type="cuda", dtype=amp, enabled=(amp != torch.float32)):
                out = model(x)
            probs = torch.softmax(out.logits.float(), dim=1)
            p_fake = fake_probability(probs, num_classes).cpu().numpy()
            pred_cls = probs.argmax(dim=1).cpu().numpy()
            for it, pf, pc in zip(buffer, p_fake, pred_cls):
                handle(it, float(pf), int(pc)); it["rgb"] = None
        except Exception as exc:
            for it in buffer:
                counts["errors"] += 1
                f_out.write(fmt_line("ER", it["truth"], "----", "error", None, None, it["key"]) + "\n")
            print(f"[GPU {device_id}] batch-error {exc!r}", file=sys.stderr, flush=True)
        counts["frames"] += len(buffer); buffer.clear()
        if args.progress_interval > 0 and time.time() - last_report[0] >= args.progress_interval:
            rate = counts["frames"] / max(time.time() - t0, 1e-9)
            print(f"[GPU {device_id}] {counts['frames']} frames  {rate:.0f}/s err={counts['errors']}", flush=True)
            last_report[0] = time.time()

    try:
        for path, kind in media_paths:
            truth = truth_of(path)
            if truth is None:
                continue
            for key, rgb in frames_of(path, kind, args.frame_stride):
                buffer.append({"key": key, "truth": truth, "rgb": rgb})
                if len(buffer) >= args.flush_size:
                    flush()
        flush()
    except Exception as exc:
        print(f"[GPU {device_id}] WORKER-ERROR {exc!r}", file=sys.stderr, flush=True)
    finally:
        f_out.close()
        result_q.put({"device_id": device_id, "tally": tally, "counts": counts})


def merge_shard_files(out_dir, basename, device_ids, header_lines):
    shards = [os.path.join(out_dir, f"results_{basename}.shard{d}.txt") for d in device_ids]
    shards = [s for s in shards if os.path.exists(s)]
    if not shards:
        return None
    dst = os.path.join(out_dir, f"results_{basename}.txt")
    with open(dst, "w") as out:
        for h in header_lines:
            out.write(h + "\n")
        for s in shards:
            out.write(open(s).read())
    for s in shards:
        try:
            os.remove(s)
        except OSError:
            pass
    return dst


def main():
    ap = argparse.ArgumentParser(description=f"{PROJECT} batch image/video tester (multi-GPU)")
    ap.add_argument("--ckpt", default="runs/gend_default/best.pt")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--input-dir", required=True)
    ap.add_argument("--out-dir", default="runs/test")
    ap.add_argument("--miss-dir", default=None)
    ap.add_argument("--devices", default="all")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--frame-stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--flush-size", type=int, default=256)
    ap.add_argument("--copy-miss", type=int, default=1)
    ap.add_argument("--progress-interval", type=float, default=30.0)
    args = ap.parse_args()

    if not os.path.isabs(args.ckpt):
        args.ckpt = os.path.join(_HERE, args.ckpt)
    if not os.path.isabs(args.config):
        args.config = os.path.join(_HERE, args.config)

    import torch
    if not torch.cuda.is_available():
        print("CUDA required.", file=sys.stderr); return 1
    if not os.path.isfile(args.ckpt):
        print(f"checkpoint not found: {args.ckpt}", file=sys.stderr); return 1
    devices, n_visible = parse_devices(args)
    bad = [d for d in devices if d < 0 or d >= n_visible]
    if not devices or bad:
        print(f"Invalid devices {bad or devices}; visible={n_visible}.", file=sys.stderr); return 1

    os.makedirs(args.out_dir, exist_ok=True)
    args.miss_dir = args.miss_dir or os.path.join(args.out_dir, "miss")

    media = [(p, k) for p, k in iter_media(args.input_dir, skip_dirs=(args.miss_dir, args.out_dir))
             if truth_of(p) is not None]
    if args.limit:
        media = media[:args.limit]
    if not media:
        print(f"No labelled image/video files under {args.input_dir}"); return 0
    n_images = sum(1 for _, k in media if k == "image")
    n_videos = sum(1 for _, k in media if k == "video")
    print(f"[{PROJECT}] media: {n_images} images + {n_videos} videos = {len(media)} files | "
          f"ckpt={args.ckpt} | GPUs={','.join(map(str, devices))}", flush=True)

    shards = [media[i::len(devices)] for i in range(len(devices))]
    ctx = mp.get_context("spawn")
    result_q = ctx.Queue()
    procs = []
    for dev, shard in zip(devices, shards):
        p = ctx.Process(target=gpu_worker, args=(dev, shard, vars(args), len(devices), result_q))
        p.start(); procs.append(p)

    tally = {}
    counts = {"frames": 0, "skipped": 0, "errors": 0, "miss_saved": 0}
    for _ in procs:
        msg = result_q.get()
        for truth, (c, n) in msg["tally"].items():
            t = tally.setdefault(truth, [0, 0]); t[0] += c; t[1] += n
        for k, v in msg["counts"].items():
            counts[k] = counts.get(k, 0) + v
        print(f"[GPU {msg['device_id']}-DONE] frames={msg['counts']['frames']} err={msg['counts']['errors']}", flush=True)
    for p in procs:
        p.join()

    col = "# columns: OK/XX/SK/ER  truth  pred  type  fake_score  match_score  image"
    tag = f"# {PROJECT} | ckpt={args.ckpt} | input={args.input_dir}"
    out_file = merge_shard_files(args.out_dir, "gend", devices, [tag, col])

    def pct(c, n):
        return 100.0 * c / n if n else float("nan")
    print(f"\n=== {PROJECT} accuracy (threshold-based; SK excluded) ===")
    recalls = []
    for truth, (c, n) in sorted(tally.items()):
        recalls.append(pct(c, n))
        print(f"  {truth:5s}: {c}/{n} = {pct(c, n):.2f}%")
    tot_c = sum(c for c, _ in tally.values()); tot_n = sum(n for _, n in tally.values())
    print(f"  OVERALL: {tot_c}/{tot_n} = {pct(tot_c, tot_n):.2f}%"
          + (f"   (balanced = {sum(recalls)/len(recalls):.2f}%)" if recalls else ""))
    print(f"  frames={counts['frames']} errors={counts['errors']} miss-saved={counts['miss_saved']}")
    print(f"\nresults -> {out_file}  | misses -> {args.miss_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
