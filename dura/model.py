from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from dura.config import ClipConfig, DURAModelConfig, LoRAConfig
from dura.modules.lora import LoRAAdapter


CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


@dataclass
class BackboneInfo:
    provider: str
    model_name: str
    output_dim: int
    load_errors: List[str]


class SmallVisionEncoder(nn.Module):
    def __init__(self, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class HashTextEncoder(nn.Module):
    def __init__(self, out_dim: int, num_buckets: int = 50000, max_chars: int = 256) -> None:
        super().__init__()
        self.num_buckets = num_buckets
        self.max_chars = max_chars
        self.embedding = nn.Embedding(num_buckets, out_dim)

    def _text_to_ids(self, text: str) -> List[int]:
        chars = list(text[: self.max_chars])
        if not chars:
            return [0]
        return [abs(hash(ch)) % self.num_buckets for ch in chars]

    def forward(self, texts: List[str], device: torch.device) -> torch.Tensor:
        feats = []
        for text in texts:
            idx = torch.tensor(self._text_to_ids(text), dtype=torch.long, device=device)
            feats.append(self.embedding(idx).mean(dim=0, keepdim=True))
        return torch.cat(feats, dim=0)


class CLIPBackbone(nn.Module):
    def __init__(self, cfg: ClipConfig, target_dim: int, logger=None) -> None:
        super().__init__()
        self.cfg = cfg
        self.target_dim = int(target_dim)
        self.logger = logger
        self.provider = ""
        self.model_name = ""
        self.output_dim = self.target_dim
        self.load_errors: List[str] = []
        self.tokenizer = None
        self.clip_model = None
        self.text_encoder = None
        self.vision_model = None
        self.text_max_length_effective = int(cfg.max_text_length)
        self.image_size_effective = int(cfg.image_size)
        self._try_load_clip()
        if not self.provider:
            self._load_fallback()
        self.text_max_length_effective = self._resolve_text_max_length()
        if self.logger:
            self.logger.info(
                "DURA backbone provider=%s model=%s output_dim=%s",
                self.provider,
                self.model_name,
                self.output_dim,
            )
            for err in self.load_errors:
                self.logger.info("DURA backbone load warning: %s", err)

    def _resolve_text_max_length(self) -> int:
        base = int(self.cfg.max_text_length)
        if self.provider != "hf_clip" or self.clip_model is None:
            return max(1, base)
        limits: List[int] = []
        tok_max = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(tok_max, int) and 1 <= tok_max < 1_000_000:
            limits.append(tok_max)
        model_cfg = getattr(self.clip_model, "config", None)
        text_cfg = getattr(model_cfg, "text_config", None) if model_cfg is not None else None
        for cfg_obj in (text_cfg, model_cfg):
            if cfg_obj is None:
                continue
            for name in ("max_position_embeddings", "max_text_position_embeddings"):
                value = getattr(cfg_obj, name, None)
                if isinstance(value, int) and value > 0:
                    limits.append(value)
        if not limits:
            return max(1, base)
        effective = max(1, min(base, min(limits)))
        if effective != base:
            self.load_errors.append(f"Adjusted text max length to model limit: requested={base}, effective={effective}")
        vision_cfg = getattr(model_cfg, "vision_config", None) if model_cfg is not None else None
        image_size = getattr(vision_cfg, "image_size", None)
        if isinstance(image_size, int) and image_size > 0:
            self.image_size_effective = int(image_size)
            if int(self.cfg.image_size) != int(image_size):
                self.load_errors.append(
                    f"Adjusted image size to model requirement: requested={int(self.cfg.image_size)}, effective={int(image_size)}"
                )
        return effective

    def _try_load_clip(self) -> None:
        if self.cfg.local_files_only:
            os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        for name in self.cfg.model_candidates:
            try:
                tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=self.cfg.local_files_only)
                model = AutoModel.from_pretrained(
                    name,
                    local_files_only=self.cfg.local_files_only,
                    weights_only=False,
                    use_safetensors=False,
                )
                if hasattr(model, "get_text_features") and hasattr(model, "get_image_features"):
                    self.provider = "hf_clip"
                    self.model_name = name
                    self.tokenizer = tokenizer
                    self.clip_model = model
                    out_dim = getattr(model.config, "projection_dim", None)
                    if out_dim is None and hasattr(model.config, "text_config"):
                        out_dim = getattr(model.config.text_config, "hidden_size", None)
                    self.output_dim = int(out_dim or self.target_dim)
                    if self.cfg.freeze_backbone:
                        for p in self.clip_model.parameters():
                            p.requires_grad = False
                    return
                self.load_errors.append(f"{name}: model does not expose get_text_features/get_image_features")
            except Exception as exc:  # noqa: BLE001
                self.load_errors.append(f"{name}: {exc}")

    def _load_fallback(self) -> None:
        self.provider = "lite_fallback"
        self.model_name = "hash_text+small_cnn"
        self.output_dim = self.target_dim
        self.text_encoder = HashTextEncoder(out_dim=self.target_dim, max_chars=max(64, self.cfg.max_text_length * 2))
        self.vision_model = SmallVisionEncoder(self.target_dim)

    def info(self) -> BackboneInfo:
        return BackboneInfo(
            provider=self.provider,
            model_name=self.model_name,
            output_dim=self.output_dim,
            load_errors=list(self.load_errors),
        )

    @staticmethod
    def _as_feature_tensor(output, name: str) -> torch.Tensor:
        if torch.is_tensor(output):
            return output
        if hasattr(output, "pooler_output") and torch.is_tensor(output.pooler_output):
            return output.pooler_output
        if hasattr(output, "last_hidden_state") and torch.is_tensor(output.last_hidden_state):
            return output.last_hidden_state[:, 0]
        if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
            return output[0]
        raise TypeError(f"Unsupported {name} feature type: {type(output)}")

    @staticmethod
    def _module_device(module: nn.Module) -> torch.device:
        try:
            return next(module.parameters()).device
        except StopIteration:
            return torch.device("cpu")

    def _clip_text(self, texts: List[str], target_device: torch.device) -> torch.Tensor:
        assert self.tokenizer is not None and self.clip_model is not None
        clip_device = self._module_device(self.clip_model)
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.text_max_length_effective,
            return_tensors="pt",
        )
        tokenized = {k: v.to(clip_device) for k, v in tokenized.items()}
        feat = self._as_feature_tensor(self.clip_model.get_text_features(**tokenized), name="text")
        return feat.to(target_device) if feat.device != target_device else feat

    def _clip_image_flat(self, images: torch.Tensor, target_device: torch.device) -> torch.Tensor:
        assert self.clip_model is not None
        clip_device = self._module_device(self.clip_model)
        x = F.interpolate(images, size=(self.image_size_effective, self.image_size_effective), mode="bilinear", align_corners=False)
        x = x.to(clip_device)
        x = (x - CLIP_MEAN.to(clip_device)) / CLIP_STD.to(clip_device)
        feat = self._as_feature_tensor(self.clip_model.get_image_features(pixel_values=x), name="image")
        return feat.to(target_device) if feat.device != target_device else feat

    def _fallback_text(self, texts: List[str], target_device: torch.device) -> torch.Tensor:
        assert self.text_encoder is not None
        enc_device = self._module_device(self.text_encoder)
        feat = self.text_encoder(texts, enc_device)
        return feat.to(target_device) if feat.device != target_device else feat

    def _fallback_image_flat(self, images: torch.Tensor) -> torch.Tensor:
        assert self.vision_model is not None
        model_device = self._module_device(self.vision_model)
        x = F.interpolate(images.to(model_device), size=(self.image_size_effective, self.image_size_effective), mode="bilinear", align_corners=False)
        feat = self.vision_model(x)
        return feat.to(images.device) if feat.device != images.device else feat

    def encode_batch(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        images = batch["images"]
        image_mask = batch["image_mask"]
        target_device = images.device
        bsz, max_images = int(images.size(0)), int(images.size(1))
        flat_images = images.reshape(bsz * max_images, images.size(2), images.size(3), images.size(4))
        flat_mask = image_mask.reshape(bsz * max_images)

        if self.provider == "hf_clip":
            text_feat = self._clip_text(batch["text"], target_device)
            flat_img_feat = self._clip_image_flat(flat_images, target_device)
        else:
            text_feat = self._fallback_text(batch["text"], target_device)
            flat_img_feat = self._fallback_image_flat(flat_images)

        flat_img_feat = flat_img_feat * flat_mask.to(flat_img_feat.device).unsqueeze(-1)
        image_feat = flat_img_feat.reshape(bsz, max_images, -1)
        return text_feat, image_feat


class ReliabilityHead(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.net(x)) + 1e-6


class DomainLoRAModel(nn.Module):
    def __init__(self, backbone: CLIPBackbone, lora_cfg: LoRAConfig, model_cfg: DURAModelConfig) -> None:
        super().__init__()
        self.backbone = backbone
        dim = backbone.output_dim
        self.text_adapter = LoRAAdapter(dim, rank=lora_cfg.rank, alpha=lora_cfg.alpha, dropout=lora_cfg.dropout)
        self.image_adapter = LoRAAdapter(dim, rank=lora_cfg.rank, alpha=lora_cfg.alpha, dropout=lora_cfg.dropout)
        self.proj = nn.Sequential(
            nn.Linear(dim * 2, model_cfg.proj_dim),
            nn.LayerNorm(model_cfg.proj_dim),
            nn.ReLU(),
            nn.Dropout(model_cfg.dropout),
        )
        self.hidden = nn.Sequential(
            nn.Linear(model_cfg.proj_dim, model_cfg.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(model_cfg.dropout),
        )
        self.classifier = nn.Linear(model_cfg.classifier_hidden_dim, 1)

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        denom = mask.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (x * mask.unsqueeze(-1)).sum(dim=1) / denom

    def encode(self, batch: Dict[str, object]) -> Tuple[torch.Tensor, torch.Tensor]:
        text_feat, image_feats = self.backbone.encode_batch(batch)
        image_feat = self._masked_mean(image_feats, batch["image_mask"].to(image_feats.device))
        return text_feat, image_feat

    def forward_encoded(self, text_feat: torch.Tensor, image_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        text_h = self.text_adapter(text_feat)
        image_h = self.image_adapter(image_feat)
        h = self.proj(torch.cat([text_h, image_h], dim=-1))
        cls_h = self.hidden(h)
        logits = self.classifier(cls_h)
        return {"logits": logits, "features": h, "cls_hidden": cls_h}

    def forward(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        if "text_feature" in batch and "image_features" in batch:
            image_feat = self._masked_mean(batch["image_features"], batch["image_mask"].to(batch["image_features"].device))
            return self.forward_encoded(batch["text_feature"], image_feat)
        text_feat, image_feat = self.encode(batch)
        return self.forward_encoded(text_feat, image_feat)

    def adapter_state(self) -> Dict[str, torch.Tensor]:
        state = {}
        for k, v in self.text_adapter.state_dict().items():
            state[f"text_adapter.{k}"] = v.detach().cpu()
        for k, v in self.image_adapter.state_dict().items():
            state[f"image_adapter.{k}"] = v.detach().cpu()
        return state

    def load_adapter_state(self, state: Dict[str, torch.Tensor]) -> None:
        text_state = {k.split("text_adapter.", 1)[1]: v for k, v in state.items() if k.startswith("text_adapter.")}
        image_state = {k.split("image_adapter.", 1)[1]: v for k, v in state.items() if k.startswith("image_adapter.")}
        self.text_adapter.load_state_dict(text_state, strict=True)
        self.image_adapter.load_state_dict(image_state, strict=True)


class DURAModel(nn.Module):
    def __init__(
        self,
        backbone: CLIPBackbone,
        lora_cfg: LoRAConfig,
        model_cfg: DURAModelConfig,
        ablation: str = "none",
        num_domains: int = 0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.ablation = str(ablation)
        self.num_domains = int(num_domains)
        self.reliability_weighting = str(model_cfg.reliability_weighting)
        self.use_adaptive_fusion = bool(model_cfg.use_adaptive_fusion)

        dim = backbone.output_dim
        post_dim = model_cfg.proj_dim
        self.shared_text_adapter = LoRAAdapter(dim, rank=lora_cfg.rank, alpha=lora_cfg.alpha, dropout=lora_cfg.dropout)
        self.shared_image_adapter = LoRAAdapter(dim, rank=lora_cfg.rank, alpha=lora_cfg.alpha, dropout=lora_cfg.dropout)
        self.private_text_adapter = LoRAAdapter(dim, rank=lora_cfg.rank, alpha=lora_cfg.alpha, dropout=lora_cfg.dropout)
        self.private_image_adapter = LoRAAdapter(dim, rank=lora_cfg.rank, alpha=lora_cfg.alpha, dropout=lora_cfg.dropout)

        self.text_rel = ReliabilityHead(dim, model_cfg.reliability_hidden_dim)
        self.shared_rel = ReliabilityHead(dim, model_cfg.reliability_hidden_dim)
        self.private_rel = ReliabilityHead(dim, model_cfg.reliability_hidden_dim)

        self.text_domain_gate = nn.Sequential(
            nn.Linear(dim * 2, model_cfg.gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(model_cfg.gate_hidden_dim, dim),
            nn.Sigmoid(),
        )
        self.visual_domain_gate = nn.Sequential(
            nn.Linear(dim * 2, model_cfg.gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(model_cfg.gate_hidden_dim, dim),
            nn.Sigmoid(),
        )
        self.modal_gate = nn.Sequential(
            nn.Linear(2, model_cfg.gate_hidden_dim),
            nn.ReLU(),
            nn.Dropout(model_cfg.dropout),
            nn.Linear(model_cfg.gate_hidden_dim, dim * 2),
            nn.Sigmoid(),
        )

        self.shared_post_proj = nn.Sequential(
            nn.Linear(dim * 2, post_dim),
            nn.LayerNorm(post_dim),
            nn.ReLU(),
            nn.Dropout(model_cfg.dropout),
        )
        self.private_post_proj = nn.Sequential(
            nn.Linear(dim * 2, post_dim),
            nn.LayerNorm(post_dim),
            nn.ReLU(),
            nn.Dropout(model_cfg.dropout),
        )
        self.cls_hidden = nn.Sequential(
            nn.Linear(dim * 2, model_cfg.classifier_hidden_dim),
            nn.ReLU(),
            nn.Dropout(model_cfg.dropout),
        )
        self.classifier = nn.Linear(model_cfg.classifier_hidden_dim, 1)
        if self.num_domains > 0:
            self.private_domain_classifier = nn.Sequential(
                nn.Linear(post_dim, model_cfg.classifier_hidden_dim),
                nn.ReLU(),
                nn.Dropout(model_cfg.dropout),
                nn.Linear(model_cfg.classifier_hidden_dim, self.num_domains),
            )
        else:
            self.private_domain_classifier = None

    def encode(self, batch: Dict[str, object]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_feat, image_feats = self.backbone.encode_batch(batch)
        return text_feat, image_feats, batch["image_mask"].to(text_feat.device)

    def load_shared_adapter_state(self, state: Dict[str, torch.Tensor]) -> None:
        text_state = {k.split("text_adapter.", 1)[1]: v for k, v in state.items() if k.startswith("text_adapter.")}
        image_state = {k.split("image_adapter.", 1)[1]: v for k, v in state.items() if k.startswith("image_adapter.")}
        self.shared_text_adapter.load_state_dict(text_state, strict=True)
        self.shared_image_adapter.load_state_dict(image_state, strict=True)

    def freeze_shared_stream(self) -> None:
        for p in self.shared_text_adapter.parameters():
            p.requires_grad = False
        for p in self.shared_image_adapter.parameters():
            p.requires_grad = False

    @staticmethod
    def _precision_from_scale(scale: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.clamp(scale ** 2, min=1e-6)

    @staticmethod
    def _global_visual_reliability(
        shared_precision: torch.Tensor,
        private_precision: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        valid_count = image_mask.float().sum(dim=1).clamp_min(1e-6)
        shared_rel = shared_precision / valid_count
        private_rel = private_precision / valid_count
        return shared_rel, private_rel, 0.5 * (shared_rel + private_rel)

    def _aggregate_visual(
        self,
        image_feats: torch.Tensor,
        image_mask: torch.Tensor,
        rel_head: ReliabilityHead,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = rel_head(image_feats).squeeze(-1)
        valid_mask = image_mask.float()
        precision = self._precision_from_scale(scale) * valid_mask
        if self.ablation == "w/o_ivw" or self.reliability_weighting == "mean":
            weights = valid_mask
        elif self.reliability_weighting == "softmax":
            logits = -scale.masked_fill(valid_mask <= 0, 1e9)
            weights = torch.softmax(logits, dim=1) * valid_mask
        else:
            weights = precision
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        summary = (image_feats * weights.unsqueeze(-1)).sum(dim=1)
        precision_sum = precision.sum(dim=1)
        return summary, precision_sum, weights, scale

    def _domain_reintegrate(
        self,
        shared_repr: torch.Tensor,
        private_repr: torch.Tensor,
        gate_layer: nn.Module,
        has_visual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.ablation == "w/o_shared":
            fused = private_repr
            gate = torch.zeros_like(private_repr)
        elif self.ablation == "w/o_private":
            fused = shared_repr
            gate = torch.zeros_like(shared_repr)
        elif self.ablation == "w/o_adaptive" or not self.use_adaptive_fusion:
            fused = 0.5 * (shared_repr + private_repr)
            gate = torch.full_like(shared_repr, 0.5)
        else:
            gate = gate_layer(torch.cat([shared_repr, private_repr], dim=-1))
            fused = shared_repr + gate * private_repr
        if has_visual is not None:
            fused = fused * has_visual.unsqueeze(-1)
            gate = gate * has_visual.unsqueeze(-1)
        return fused, gate

    def forward_encoded(
        self,
        text_feat: torch.Tensor,
        image_feats: torch.Tensor,
        image_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        shared_text = self.shared_text_adapter(text_feat)
        private_text = self.private_text_adapter(text_feat)
        shared_images = self.shared_image_adapter(image_feats)
        private_images = self.private_image_adapter(image_feats)

        shared_visual, shared_precision, shared_weights, shared_scale = self._aggregate_visual(
            shared_images,
            image_mask,
            self.shared_rel,
        )
        private_visual, private_precision, private_weights, private_scale = self._aggregate_visual(
            private_images,
            image_mask,
            self.private_rel,
        )
        has_visual = (image_mask.sum(dim=1) > 0).float()
        shared_visual_rel, private_visual_rel, global_visual_rel = self._global_visual_reliability(
            shared_precision=shared_precision,
            private_precision=private_precision,
            image_mask=image_mask,
        )
        text_scale = self.text_rel(shared_text).squeeze(-1)
        text_precision = self._precision_from_scale(text_scale)

        fused_text, text_gate = self._domain_reintegrate(shared_text, private_text, self.text_domain_gate)
        fused_visual, visual_gate = self._domain_reintegrate(shared_visual, private_visual, self.visual_domain_gate, has_visual=has_visual)

        if self.ablation == "w/o_adaptive" or not self.use_adaptive_fusion:
            modal_gate = torch.full((text_feat.size(0), fused_text.size(-1) * 2), 0.5, dtype=text_feat.dtype, device=text_feat.device)
        else:
            modal_gate = self.modal_gate(torch.stack([text_precision, global_visual_rel], dim=-1))
        raw_final = torch.cat([fused_text, fused_visual], dim=-1)
        h_final = raw_final * modal_gate

        z_shared = self.shared_post_proj(torch.cat([shared_text, shared_visual * has_visual.unsqueeze(-1)], dim=-1))
        z_private = self.private_post_proj(torch.cat([private_text, private_visual * has_visual.unsqueeze(-1)], dim=-1))
        if self.ablation == "w/o_shared":
            z_shared = torch.zeros_like(z_shared)
        if self.ablation == "w/o_private":
            z_private = torch.zeros_like(z_private)

        hidden = self.cls_hidden(h_final)
        logits = self.classifier(hidden)
        private_domain_logits = self.private_domain_classifier(z_private) if self.private_domain_classifier is not None else None

        return {
            "logits": logits,
            "private_domain_logits": private_domain_logits,
            "z_shared": z_shared,
            "z_private": z_private,
            "h_final": h_final,
            "cls_hidden": hidden,
            "shared_visual": shared_visual,
            "private_visual": private_visual,
            "shared_weights": shared_weights,
            "private_weights": private_weights,
            "shared_visual_reliability": shared_visual_rel,
            "private_visual_reliability": private_visual_rel,
            "global_visual_reliability": global_visual_rel,
            "text_reliability": text_precision,
            "has_visual": has_visual,
            "text_domain_gate": text_gate,
            "visual_domain_gate": visual_gate,
            "modal_gate": modal_gate,
            "shared_visual_scale": shared_scale,
            "private_visual_scale": private_scale,
            "text_scale": text_scale,
        }

    def forward(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        if "text_feature" in batch and "image_features" in batch:
            return self.forward_encoded(
                text_feat=batch["text_feature"],
                image_feats=batch["image_features"],
                image_mask=batch["image_mask"].to(batch["text_feature"].device),
            )
        text_feat, image_feats, image_mask = self.encode(batch)
        return self.forward_encoded(text_feat, image_feats, image_mask)
