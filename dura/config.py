from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class PathConfig:
    dataset_root: str = ""
    output_root: str = "outputs"
    clean_snapshot_root: str = "outputs/data_snapshots"
    split_map_path: str = ""
    cache_dir: str = ""


@dataclass
class DataConfig:
    dataset_name: str = "weibo21"  # weibo21 | weibo | gossipcop
    weibo_tweets_dir: str = "tweets"
    weibo_train_nonrumor_file: str = "train_nonrumor.txt"
    weibo_train_rumor_file: str = "train_rumor.txt"
    weibo_test_nonrumor_file: str = "test_nonrumor.txt"
    weibo_test_rumor_file: str = "test_rumor.txt"
    weibo_rumor_image_dir: str = "rumor_images"
    weibo_nonrumor_image_dir: str = "nonrumor_images"
    weibo_train_split_pickle: str = "train_id.pickle"
    weibo_val_split_pickle: str = "validate_id.pickle"
    weibo_test_split_pickle: str = "test_id.pickle"
    weibo_domain_map_json: str = ""
    gossipcop_real_dir: str = "real"
    gossipcop_fake_dir: str = "fake"
    gossipcop_news_json: str = "news_content.json"
    gossipcop_images_dirname: str = "images"
    fake_json: str = "origin/fake_release_all.json"
    real_json: str = "origin/real_release_all.json"
    image_dirs: List[str] = field(default_factory=lambda: ["origin/fake", "origin/real", "image"])
    deduplicate_by_raw_id: bool = False
    duplicate_policy: str = "keep_first"
    include_comments: bool = False
    missing_image_strategy: str = "mask"
    max_text_chars: int = 512
    max_train_samples: int = 0
    max_eval_samples: int = 0
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    split_seed: int = 42
    save_split_map: bool = True
    sampler: str = "none"
    batch_size: int = 64
    eval_batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    non_blocking_transfer: bool = True
    pretokenize_text: bool = True
    use_feature_cache: bool = True
    build_feature_cache_if_missing: bool = True
    force_rebuild_feature_cache: bool = False
    feature_cache_batch_size: int = 128
    max_images_per_post: int = 8
    min_images_per_post: int = 0


@dataclass
class ClipConfig:
    model_candidates: List[str] = field(
        default_factory=lambda: [
            "OFA-Sys/chinese-clip-vit-base-patch16",
            "openai/clip-vit-base-patch16",
            "openai/clip-vit-base-patch32",
        ]
    )
    local_files_only: bool = True
    max_text_length: int = 128
    image_size: int = 224
    freeze_backbone: bool = True
    fallback_text_model: str = "hfl/chinese-macbert-large"


@dataclass
class LoRAConfig:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.1


@dataclass
class DURAModelConfig:
    proj_dim: int = 512
    fusion_hidden_dim: int = 512
    classifier_hidden_dim: int = 256
    dropout: float = 0.2
    use_adaptive_fusion: bool = True
    reliability_hidden_dim: int = 256
    gate_hidden_dim: int = 256
    reliability_weighting: str = "inverse_variance"  # inverse_variance | softmax | mean


@dataclass
class LossConfig:
    classification: str = "ce"  # ce | focal
    class_weighting: str = "inverse"  # none | inverse | sqrt_inverse
    focal_alpha_pos: float = 0.5
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    orth_weight: float = 0.05
    reliability_weight: float = 0.0
    reliability_eps: float = 1e-6
    private_domain_weight: float = 0.0


@dataclass
class OptimConfig:
    lr: float = 1.5e-4
    min_lr: float = 1e-6
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    grad_clip_norm: float = 1.0


@dataclass
class TrainConfig:
    stage1_epochs: int = 6
    stage3_epochs: int = 15
    early_stopping_patience: int = 3
    monitor_metric: str = "balanced_acc"
    monitor_mode: str = "max"
    orth_warmup_epochs: int = 0
    orth_warmup_init_factor: float = 0.0
    ema_enabled: bool = False
    ema_decay: float = 0.999
    amp: bool = False
    log_every_steps: int = 20
    max_steps_per_epoch: int = 0
    show_progress: bool = True


@dataclass
class DareConfig:
    method: str = "dare"  # dare | avg | ties
    drop_rate: float = 0.9
    merge_weights: str = "equal"  # equal | domain_size
    seed: int = 42


@dataclass
class EvalConfig:
    threshold_metric: str = "balanced_acc"  # f1 | macro_f1 | balanced_acc | real_f1 | fake_f1 | acc_macro_f1
    threshold_min: float = 0.05
    threshold_max: float = 0.95
    threshold_step: float = 0.01
    save_logits: bool = True


@dataclass
class VizConfig:
    tsne_perplexity: int = 30
    tsne_random_state: int = 42
    tsne_max_points: int = 2000


@dataclass
class ExperimentConfig:
    name: str = ""
    mode: str = "all-domain"
    ablation: str = "none"  # none | w/o_shared | w/o_private | w/o_orth | w/o_ivw | w/o_adaptive
    seed: int = 42
    device: str = "cuda"
    enable_tensorboard: bool = True
    save_predictions: bool = True
    data_parallel: bool = False
    gpu_ids: List[int] = field(default_factory=list)


@dataclass
class DURAConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    clip: ClipConfig = field(default_factory=ClipConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    model: DURAModelConfig = field(default_factory=DURAModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    dare: DareConfig = field(default_factory=DareConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    viz: VizConfig = field(default_factory=VizConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)


T = TypeVar("T")


def _coerce_value(field_type: Any, value: Any) -> Any:
    origin = get_origin(field_type)

    if origin is Union:
        union_types = [x for x in get_args(field_type) if x is not type(None)]
        if len(union_types) == 1:
            return _coerce_value(union_types[0], value)
        return value

    if origin in (list, List):
        inner = get_args(field_type)[0]
        if not isinstance(value, list):
            raise ConfigError(f"Expected list, got {type(value)}")
        return [_coerce_value(inner, x) for x in value]

    if isinstance(field_type, type) and is_dataclass(field_type):
        if not isinstance(value, dict):
            raise ConfigError(f"Expected dict for dataclass {field_type}, got {type(value)}")
        return _from_dict(field_type, value)

    if field_type is bool and isinstance(value, str):
        val = value.lower().strip()
        if val in {"1", "true", "yes", "y"}:
            return True
        if val in {"0", "false", "no", "n"}:
            return False
        raise ConfigError(f"Cannot parse bool from {value}")

    if field_type in (int, float, str, bool):
        try:
            return field_type(value)
        except Exception as exc:
            raise ConfigError(f"Cannot cast value {value!r} to {field_type}") from exc

    return value


def _from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    kwargs: Dict[str, Any] = {}
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data.keys()) - known)
    if unknown:
        raise ConfigError(f"Unknown fields for {cls.__name__}: {unknown}")

    type_hints = get_type_hints(cls)
    for f in fields(cls):
        field_type = type_hints.get(f.name, f.type)
        if f.name in data:
            kwargs[f.name] = _coerce_value(field_type, data[f.name])
        elif f.default is not MISSING:
            kwargs[f.name] = f.default
        elif f.default_factory is not MISSING:  # type: ignore[attr-defined]
            kwargs[f.name] = f.default_factory()  # type: ignore[misc]
        else:
            raise ConfigError(f"Missing required field {cls.__name__}.{f.name}")
    return cls(**kwargs)


def _set_nested(cfg_dict: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = cfg_dict
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    cur[parts[-1]] = value


def _parse_override_value(raw: str) -> Any:
    lowered = raw.lower().strip()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered == "null":
        return None
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        pass
    if raw.startswith("[") or raw.startswith("{"):
        return yaml.safe_load(raw)
    return raw


def apply_overrides(raw_cfg: Dict[str, Any], overrides: Optional[List[str]]) -> Dict[str, Any]:
    if not overrides:
        return raw_cfg
    cfg = dict(raw_cfg)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(f"Override must be key=value, got {item}")
        key, value = item.split("=", 1)
        _set_nested(cfg, key.strip(), _parse_override_value(value.strip()))
    return cfg


def validate_config(cfg: DURAConfig) -> None:
    valid_metric_names = {"acc", "precision", "recall", "f1", "macro_f1", "balanced_acc", "real_f1", "fake_f1", "acc_macro_f1"}
    valid_modes = {"all-domain"}
    valid_ablation = {"none", "w/o_shared", "w/o_private", "w/o_orth", "w/o_ivw", "w/o_adaptive"}

    if cfg.experiment.mode not in valid_modes:
        raise ConfigError(f"Invalid experiment.mode={cfg.experiment.mode}, expected one of {sorted(valid_modes)}")
    if cfg.experiment.ablation not in valid_ablation:
        raise ConfigError(f"Invalid experiment.ablation={cfg.experiment.ablation}, expected one of {sorted(valid_ablation)}")
    if cfg.data.dataset_name not in {"weibo21", "weibo", "gossipcop"}:
        raise ConfigError("data.dataset_name must be one of weibo21/weibo/gossipcop")
    if cfg.data.duplicate_policy not in {"keep_first", "keep_last"}:
        raise ConfigError("data.duplicate_policy must be keep_first or keep_last")
    if cfg.data.missing_image_strategy not in {"mask", "zero"}:
        raise ConfigError("data.missing_image_strategy must be mask or zero")
    if cfg.data.sampler not in {"none", "class", "domain", "class_domain"}:
        raise ConfigError("data.sampler must be one of none/class/domain/class_domain")
    if cfg.data.num_workers < 0:
        raise ConfigError("data.num_workers must be >= 0")
    if cfg.data.prefetch_factor <= 0:
        raise ConfigError("data.prefetch_factor must be > 0")
    if cfg.data.num_workers == 0 and cfg.data.persistent_workers:
        raise ConfigError("data.persistent_workers=true requires data.num_workers > 0")
    if cfg.data.feature_cache_batch_size <= 0:
        raise ConfigError("data.feature_cache_batch_size must be > 0")
    if cfg.data.max_images_per_post <= 0:
        raise ConfigError("data.max_images_per_post must be > 0")

    if cfg.loss.classification not in {"ce", "focal"}:
        raise ConfigError("loss.classification must be ce or focal")
    if cfg.loss.class_weighting not in {"none", "inverse", "sqrt_inverse"}:
        raise ConfigError("loss.class_weighting must be one of none/inverse/sqrt_inverse")
    if cfg.loss.orth_weight < 0.0:
        raise ConfigError("loss.orth_weight must be >= 0")
    if cfg.loss.reliability_weight < 0.0:
        raise ConfigError("loss.reliability_weight must be >= 0")
    if cfg.loss.reliability_eps <= 0.0:
        raise ConfigError("loss.reliability_eps must be > 0")
    if cfg.loss.private_domain_weight < 0.0:
        raise ConfigError("loss.private_domain_weight must be >= 0")

    if not (0.0 <= cfg.data.val_ratio < 1.0):
        raise ConfigError("data.val_ratio must be in [0,1)")
    if not (0.0 <= cfg.data.test_ratio < 1.0):
        raise ConfigError("data.test_ratio must be in [0,1)")
    if cfg.data.val_ratio + cfg.data.test_ratio >= 1.0:
        raise ConfigError("val_ratio + test_ratio must be < 1.0 in all-domain mode")

    if cfg.dare.method not in {"dare", "avg", "ties"}:
        raise ConfigError("dare.method must be dare, avg, or ties")
    if not (0.0 <= cfg.dare.drop_rate < 1.0):
        raise ConfigError("dare.drop_rate must be in [0,1)")
    if cfg.dare.merge_weights not in {"equal", "domain_size"}:
        raise ConfigError("dare.merge_weights must be equal or domain_size")

    if cfg.model.reliability_weighting not in {"inverse_variance", "softmax", "mean"}:
        raise ConfigError("model.reliability_weighting must be inverse_variance/softmax/mean")
    if cfg.model.reliability_hidden_dim <= 0 or cfg.model.gate_hidden_dim <= 0:
        raise ConfigError("model reliability/gate hidden dims must be > 0")
    if cfg.lora.rank <= 0:
        raise ConfigError("lora.rank must be > 0")

    if cfg.train.monitor_mode not in {"max", "min"}:
        raise ConfigError("train.monitor_mode must be max or min")
    if cfg.train.monitor_metric not in valid_metric_names:
        raise ConfigError(f"train.monitor_metric must be one of {sorted(valid_metric_names)}")
    if cfg.eval.threshold_metric not in valid_metric_names:
        raise ConfigError(f"eval.threshold_metric must be one of {sorted(valid_metric_names)}")
    if cfg.train.orth_warmup_epochs < 0:
        raise ConfigError("train.orth_warmup_epochs must be >= 0")
    if not (0.0 <= cfg.train.orth_warmup_init_factor <= 1.0):
        raise ConfigError("train.orth_warmup_init_factor must be in [0,1]")
    if not (0.0 < cfg.train.ema_decay <= 1.0):
        raise ConfigError("train.ema_decay must be in (0,1]")

    if cfg.experiment.data_parallel and not cfg.experiment.device.startswith("cuda"):
        raise ConfigError("experiment.data_parallel=true requires experiment.device to be cuda*")
    if any(i < 0 for i in cfg.experiment.gpu_ids):
        raise ConfigError("experiment.gpu_ids must be non-negative")
    if len(set(cfg.experiment.gpu_ids)) != len(cfg.experiment.gpu_ids):
        raise ConfigError("experiment.gpu_ids must not contain duplicates")


def private_domain_loss_enabled(cfg: DURAConfig) -> bool:
    return float(cfg.loss.private_domain_weight) > 0.0


def load_config(config_path: Union[str, Path], overrides: Optional[List[str]] = None) -> DURAConfig:
    path = Path(config_path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Config root must be a mapping")
    raw = apply_overrides(raw, overrides)
    cfg = _from_dict(DURAConfig, raw)
    validate_config(cfg)
    return cfg


def to_dict(cfg: DURAConfig) -> Dict[str, Any]:
    return asdict(cfg)


def load_config_from_dict(raw_cfg: Dict[str, Any]) -> DURAConfig:
    if not isinstance(raw_cfg, dict):
        raise ConfigError("Config root must be a mapping")
    cfg = _from_dict(DURAConfig, raw_cfg)
    validate_config(cfg)
    return cfg
