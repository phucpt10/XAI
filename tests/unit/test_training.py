import random

import numpy as np
import pytest

from plantxai_stability.training import _restore_rng_state, _validate_resume_payload


def _config() -> dict[str, object]:
    return {
        "num_classes": 5,
        "config_hash": "protocol-hash",
        "manifest_sha256": "manifest-hash",
        "pretrained": True,
        "pretrained_weights": {
            "resnet50": "IMAGENET1K_V2",
            "efficientnet_b0": "IMAGENET1K_V1",
        },
        "fine_tuning": "full_model",
        "loss": "cross_entropy",
        "class_weighting": "none",
        "max_epochs": 50,
        "batch_size": 32,
        "learning_rate": 0.001,
        "optimizer": "adamw",
        "weight_decay": 0.0001,
        "scheduler": "cosine",
        "early_stopping_patience": 8,
        "selection_metric": "validation_macro_f1",
        "mixed_precision": True,
        "deterministic_algorithms": True,
        "seed": 42,
        "software_version": "0.1.0",
        "git_commit": "abc123",
    }


def _payload() -> dict[str, object]:
    config = _config()
    return {
        "format_version": 2,
        "model_id": "resnet50",
        "num_classes": 5,
        "protocol_hash": "protocol-hash",
        "manifest_sha256": "manifest-hash",
        "training_config": {
            key: config[key]
            for key in (
                "pretrained",
                "pretrained_weights",
                "fine_tuning",
                "loss",
                "class_weighting",
                "max_epochs",
                "batch_size",
                "learning_rate",
                "optimizer",
                "weight_decay",
                "scheduler",
                "early_stopping_patience",
                "selection_metric",
                "mixed_precision",
                "deterministic_algorithms",
                "seed",
                "software_version",
                "git_commit",
            )
        },
    }


def test_resume_lineage_accepts_exact_match() -> None:
    _validate_resume_payload(_payload(), "resnet50", _config())


def test_resume_lineage_rejects_manifest_change() -> None:
    payload = _payload()
    payload["manifest_sha256"] = "changed"
    with pytest.raises(ValueError, match="manifest_sha256"):
        _validate_resume_payload(payload, "resnet50", _config())


def test_resume_lineage_rejects_training_config_change() -> None:
    config = _config()
    config["learning_rate"] = 0.01
    with pytest.raises(ValueError, match="training_config"):
        _validate_resume_payload(_payload(), "resnet50", config)


def test_restore_rng_state_moves_byte_tensors_back_to_cpu() -> None:
    class FakeTensor:
        def __init__(self, device: str = "cuda") -> None:
            self.device = device
            self.dtype = "uint8"
            self.ndim = 1

        def detach(self) -> "FakeTensor":
            return self

        def to(self, *, device: str, dtype: str) -> "FakeTensor":
            self.device = device
            self.dtype = dtype
            return self

        def contiguous(self) -> "FakeTensor":
            return self

        def numel(self) -> int:
            return 16

    class FakeCuda:
        restored: list[FakeTensor] = []

        @staticmethod
        def is_available() -> bool:
            return True

        @classmethod
        def set_rng_state_all(cls, values: list[FakeTensor]) -> None:
            assert all(value.device == "cpu" for value in values)
            cls.restored = values

    class FakeTorch:
        Tensor = FakeTensor
        uint8 = "uint8"
        cuda = FakeCuda
        restored: FakeTensor | None = None

        @classmethod
        def set_rng_state(cls, value: FakeTensor) -> None:
            assert value.device == "cpu"
            cls.restored = value

    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": FakeTensor(),
        "torch_cuda": [FakeTensor()],
    }
    _restore_rng_state(state, FakeTorch)
    assert FakeTorch.restored is not None
    assert len(FakeCuda.restored) == 1
