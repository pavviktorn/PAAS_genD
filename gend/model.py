"""GenD model — faithful port of the official repo (yermandy/GenD).

- Encoder: CLIP vision tower; feature = `vision_model(x).pooler_output` (post-LN
  [CLS], dim 1024 for ViT-L/14). visual_projection is kept for state-dict parity
  with the released checkpoint but is NOT used in the forward (matches upstream).
- Head: LinearProbe. The l2_embeddings (F.normalize of the feature) are always
  produced for the uniformity/alignment losses; with head="LinearNorm" the linear
  classifier also consumes the L2-normalized feature (the hyperspherical setup).
- Trainable: only the LayerNorm groups of the backbone
  (["pre_layrnorm","layer_norm1","layer_norm2","post_layernorm"]) + the head.
  Everything else in CLIP is frozen (GenD's parameter-efficient adaptation).
- Init: the released GenD checkpoint's `feature_extractor.*` weights (finetuned
  LN baked into a full CLIP vision tower) are loaded; the 2-class head is dropped
  in favour of a fresh `num_classes` head.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

DEFAULT_UNFREEZE = ["pre_layrnorm", "layer_norm1", "layer_norm2", "post_layernorm"]


@dataclass
class HeadOutput:
    logits: torch.Tensor
    l2_embeddings: torch.Tensor


class CLIPEncoder(nn.Module):
    def __init__(self, base_name="openai/clip-vit-large-patch14"):
        super().__init__()
        clip = CLIPModel.from_pretrained(base_name)
        self.vision_model = clip.vision_model
        self.visual_projection = clip.visual_projection  # unused in forward; kept for parity
        self.features_dim = self.vision_model.config.hidden_size

    def forward(self, pixel_values):
        return self.vision_model(pixel_values).pooler_output  # (B, 1024)

    def get_features_dim(self):
        return self.features_dim


class LinearProbe(nn.Module):
    def __init__(self, input_dim, num_classes, normalize_inputs=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.normalize_inputs = normalize_inputs

    def forward(self, x):
        l2 = F.normalize(x, p=2, dim=1)
        logits = self.linear(l2 if self.normalize_inputs else x)
        return HeadOutput(logits=logits, l2_embeddings=l2)


class GenDModel(nn.Module):
    def __init__(self, base_name="openai/clip-vit-large-patch14", num_classes=3,
                 head="LinearNorm", unfreeze_layers=None):
        super().__init__()
        self.feature_extractor = CLIPEncoder(base_name)
        dim = self.feature_extractor.get_features_dim()
        self.num_classes = num_classes
        self.model = LinearProbe(dim, num_classes, normalize_inputs=(head == "LinearNorm"))
        self.unfreeze_layers = DEFAULT_UNFREEZE if unfreeze_layers is None else unfreeze_layers
        self._freeze()

    def _freeze(self):
        # freeze the whole backbone, then unfreeze only the named LayerNorm groups
        self.feature_extractor.requires_grad_(False)
        for name, p in self.feature_extractor.named_parameters():
            if any(layer in name for layer in self.unfreeze_layers):
                p.requires_grad_(True)
        # head is always trainable
        for p in self.model.parameters():
            p.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        # keep frozen backbone modules in eval so their (frozen) dropout is off;
        # trainable LayerNorm has no train/eval-dependent behaviour, so this is safe.
        self.feature_extractor.eval()
        return self

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(self, pixel_values) -> HeadOutput:
        feats = self.feature_extractor(pixel_values)
        return self.model(feats)

    # ---- init from released GenD checkpoint (feature_extractor.* subset) ----
    def load_pretrained_encoder(self, safetensors_path):
        from safetensors.torch import load_file
        sd = load_file(safetensors_path)
        enc_sd = {k[len("feature_extractor."):]: v
                  for k, v in sd.items() if k.startswith("feature_extractor.")}
        missing, unexpected = self.feature_extractor.load_state_dict(enc_sd, strict=False)
        return len(enc_sd), missing, unexpected

    # ---- checkpoint I/O: save only the trainable subset (LN + head) ----
    def export_state(self):
        trainable = {n: p.detach().cpu() for n, p in self.named_parameters() if p.requires_grad}
        return {"trainable": trainable,
                "config": {"num_classes": self.num_classes,
                           "normalize_inputs": self.model.normalize_inputs,
                           "unfreeze_layers": self.unfreeze_layers}}

    def load_trainable(self, state):
        own = dict(self.named_parameters())
        with torch.no_grad():
            for n, v in state["trainable"].items():
                if n in own:
                    own[n].copy_(v.to(own[n].device, own[n].dtype))
