"""Typed configuration for the active concept-retrofit experiments."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    name: str
    revision: str | None = None
    trainable_top_layers: int = 0
    use_lora: bool = False
    lora_rank: int = 16

    def validate(self) -> None:
        if self.trainable_top_layers < 0:
            raise ValueError("trainable_top_layers must be non-negative")
        if self.use_lora and self.trainable_top_layers == 0:
            raise ValueError("LoRA requires at least one trainable top layer")
        if self.lora_rank <= 0:
            raise ValueError("lora_rank must be positive")


@dataclass(frozen=True)
class BottleneckConfig:
    known_concepts: int
    unknown_features: int
    unknown_rank: int
    known_topk: int
    unknown_topk: int
    residual_schedule: str = "linear_warmup"

    def validate(self) -> None:
        if self.known_concepts <= 0 or self.unknown_features <= 0:
            raise ValueError("known_concepts and unknown_features must be positive")
        if not 0 < self.known_topk <= self.known_concepts:
            raise ValueError("known_topk must be in [1, known_concepts]")
        if not 0 < self.unknown_topk <= self.unknown_features:
            raise ValueError("unknown_topk must be in [1, unknown_features]")
        if not 0 < self.unknown_rank <= self.unknown_features:
            raise ValueError("unknown_rank must be in [1, unknown_features]")


@dataclass(frozen=True)
class LossConfig:
    next_token_ce: float = 1.0
    teacher_kl: float = 1.0
    concept_pu: float = 1.0
    reconstruction: float = 1.0
    residual: float = 0.0
    sparsity: float = 0.0
    leakage: float = 0.0

    def validate(self) -> None:
        if any(value < 0 for value in asdict(self).values()):
            raise ValueError("loss weights must be non-negative")
        if self.next_token_ce == 0 and self.teacher_kl == 0:
            raise ValueError("at least one language-preservation loss must be enabled")


@dataclass(frozen=True)
class DataConfig:
    train_manifest: str
    max_tokens: int
    streaming: bool = True
    token_level_labels: bool = False

    def validate(self) -> None:
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: ModelConfig
    bottleneck: BottleneckConfig
    losses: LossConfig
    data: DataConfig
    seed: int = 1729
    notes: str = ""

    def validate(self) -> None:
        self.model.validate()
        self.bottleneck.validate()
        self.losses.validate()
        self.data.validate()
        if self.model.trainable_top_layers == 0 and self.losses.leakage > 0:
            raise ValueError("leakage minimization belongs in an adapting-backbone experiment")


def load_config(path: str | Path) -> ExperimentConfig:
    raw = json.loads(Path(path).read_text())
    config = ExperimentConfig(
        name=raw["name"], model=ModelConfig(**raw["model"]),
        bottleneck=BottleneckConfig(**raw["bottleneck"]),
        losses=LossConfig(**raw["losses"]), data=DataConfig(**raw["data"]),
        seed=raw.get("seed", 1729), notes=raw.get("notes", ""),
    )
    config.validate()
    return config
