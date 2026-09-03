# PAAS_genD — GenD on MIDS

Standalone implementation of **GenD** (*Deepfake Detection that Generalizes Across
Benchmarks*, WACV 2026, arXiv:2508.06248), faithfully following the official code
(`reference_repos/GenD`), trained/evaluated on the MIDS dataset.

## Method (exactly as in the paper's best config `wacv-LN+L2+UnAl`)

- **Backbone**: CLIP ViT-L/14 (`openai/clip-vit-large-patch14`), feature =
  `vision_model(x).pooler_output` (post-LN [CLS], 1024-d). **Frozen** except the
  LayerNorm affine params.
- **Parameter-efficient adaptation**: fine-tune **only the LayerNorm groups**
  `["pre_layrnorm","layer_norm1","layer_norm2","post_layernorm"]` + a linear head
  → **0.1055M / 304M = 0.035%** trainable (the paper's ~0.03%).
- **Hyperspherical head** (`LinearNorm`): the 1024-d feature is L2-normalized, then
  a linear layer produces `num_classes` logits.
- **Loss** = `CE(1.0) + uniformity(0.5) + alignment(0.1)` on the unit-normalized
  embeddings (metric learning on the sphere), ported verbatim from `src/losses`.
- **Init**: the released `yermandy/GenD_CLIP_L_14` encoder (`feature_extractor.*`,
  finetuned LN baked into a full CLIP vision tower) is loaded; a fresh 3-class head
  is trained. (The pretrained 2-class head is discarded.)
- **Input**: FULL image, no face cropping — CLIP preprocessing (resize shortest
  edge to 224 bicubic, center-crop 224, CLIP normalize), i.e. the released model's
  `feature_extractor.preprocess`.

## Labels — 3-class (real/pad/deepfake), like GSD

Path-derived via vendored `get_label_all` (REAL=0, PAD=1, DEEPFAKE=2, MAKEUP=3,
UNKNOWN=-1) → mapped to **real=0 / pad=1 / deepfake=2** (MAKEUP→pad; UNKNOWN
dropped). `num_classes: 2` collapses to real/fake. Inverse-frequency CE weights on.

## Metrics — primary AUC

Headline **`bin_auc`** (binary real-vs-fake AUROC, fake=positive) drives `best.pt`.
Also reported: `ovr_auroc` (3-class one-vs-rest macro AUROC, GenD's own metric),
`ap`, `acc`, `eer`, per-class recall, and `fake_recall@real95` operating point.

## Data / environment

- train: `/datasets/work/vLLM/temp/testset/testset_mids/mids_first_half.json` (1.71M)
- val:   `/datasets/work/vLLM/temp/testset/testset_mids/mids_testset.json` (30k)
- `python3.12` (torch 2.11+cu130, transformers 4.37.2). Slim index cached at
  `<json>.gend_idx.3c.tsv`.
- **Standalone / offline**: the base CLIP-L/14 and the pretrained GenD checkpoint
  are vendored in-repo, so no network is needed:
  - `base_models/clip-vit-large-patch14/` (base CLIP, 224px)
  - `weights/GenD_CLIP_L_14.safetensors` (pretrained init; only `feature_extractor.*` is used)
  The launcher exports `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`.
  Runs at **224px** to match the pretrained model (its position embeddings are fixed
  to 224; use 336 only without this init, or with pos-embedding interpolation).

## Train (GPUs 0,1,2,3, ~50% CPU)

```bash
cd /datasets/work/vLLM/temp/PAAS_genD
GPUS=0,1,2,3 OMP_NUM_THREADS=8 ./run_train.sh
```

Defaults (best model): AdamW lr 3e-4 → cosine → 1e-5 (warmup 500), wd 0,
batch 128/GPU, bf16, 2 epochs, eval every 5000 steps, select by `bin_auc`.
Override any from CLI, e.g. `./run_train.sh --epochs 3 --batch_size 96`.
Checkpoints (`best.pt`/`last.pt`, ~0.45 MB — LN+head only), `metrics_log.jsonl`
in `out_dir` (`runs/gend_default`).

## Inference

```bash
python3.12 infer.py --ckpt runs/gend_default/best.pt --image /path/face.jpg   # single image
python3.12 infer.py --ckpt runs/gend_default/best.pt --eval                    # full testset report
python3.12 test_video_image_batch.py --ckpt runs/gend_default/best.pt \
    --input-dir /datasets/work/vLLM/temp/testset --devices 0,1,2,3             # batch folder -> results_gend.txt
```

Checkpoints store only the trainable LN+head; the frozen backbone is rebuilt from
`clip_base` and the finetuned LN/head loaded on top. `results_gend.txt` uses the
same unified line format as the other PAAS testers.

## Files

```
config.json                 # defaults (best GenD config)
run_train.sh                # DDP launcher
train.py                    # DDP training
infer.py                    # single-image + full-eval
test_video_image_batch.py   # multi-GPU folder tester -> results_gend.txt
gend/model.py               # CLIP encoder + LinearNorm head + LN-only unfreeze + pretrained init
gend/losses.py              # CE + uniformity + alignment (ported from official code)
gend/data.py                # MIDS json, get_label_all 3-class, CLIP-224 transforms + GenD augs
gend/metrics.py             # bin_auc / ovr_auroc / ap / per-class recall / floors
gend/engine.py              # distributed eval
gend/get_label.py           # vendored path->label
gend/utils.py               # dist / seeding / CPU cap
```
