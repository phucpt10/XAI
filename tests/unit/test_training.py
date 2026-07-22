import pytest

from plantxai_stability.training import _validate_resume_payload


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
