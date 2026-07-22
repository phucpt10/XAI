from pathlib import Path
from typing import Any

import pytest

from plantxai_stability import training


def test_training_writes_resumable_and_lineage_bound_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")

    class TinyWrapper:
        def __init__(
            self, model_id: str, num_classes: int = 2, pretrained: bool = False
        ) -> None:
            self.model_id = model_id
            self.num_classes = num_classes
            self.pretrained = pretrained
            self.model = torch.nn.Sequential(
                torch.nn.Flatten(), torch.nn.Linear(3 * 2 * 2, num_classes)
            )

    monkeypatch.setattr(training, "ModelWrapper", TinyWrapper)
    batches: list[dict[str, Any]] = [
        {
            "model_tensor": torch.tensor(
                [
                    [[[0.0, 0.0], [0.0, 0.0]]] * 3,
                    [[[1.0, 1.0], [1.0, 1.0]]] * 3,
                ],
                dtype=torch.float32,
            ),
            "label": torch.tensor([0, 1], dtype=torch.long),
        }
    ]
    config = {
        "num_classes": 2,
        "class_names": ["a", "b"],
        "pretrained": False,
        "pretrained_weights": {
            "resnet50": "IMAGENET1K_V2",
            "efficientnet_b0": "IMAGENET1K_V1",
        },
        "fine_tuning": "full_model",
        "loss": "cross_entropy",
        "class_weighting": "none",
        "max_epochs": 2,
        "batch_size": 2,
        "learning_rate": 0.01,
        "optimizer": "adamw",
        "weight_decay": 0.0,
        "scheduler": "cosine",
        "early_stopping_patience": 2,
        "selection_metric": "validation_macro_f1",
        "mixed_precision": False,
        "deterministic_algorithms": True,
        "seed": 42,
        "device": "cpu",
        "config_hash": "protocol-hash",
        "manifest_sha256": "manifest-hash",
        "software_version": "0.1.0",
        "git_commit": "abc123",
    }
    evidence = training.train_model(
        "tiny", batches, batches, config, tmp_path / "run"
    )
    assert Path(evidence.checkpoint_path).is_file()
    assert Path(evidence.latest_checkpoint_path).is_file()
    assert Path(evidence.history_path).is_file()
    assert evidence.epochs_completed == 2
    wrapper = TinyWrapper("tiny", 2, False)
    payload = training.load_checkpoint(
        wrapper,
        evidence.checkpoint_path,
        expected_protocol_hash="protocol-hash",
        expected_manifest_sha256="manifest-hash",
    )
    assert payload["checkpoint_role"] == "validation_selected_best"
    assert payload["test_split_accessed"] is False

    with pytest.raises(FileExistsError, match="Training output exists"):
        training.train_model("tiny", batches, batches, config, tmp_path / "run")

    resumed = training.train_model(
        "tiny", batches, batches, config, tmp_path / "run", resume=True
    )
    assert resumed.checkpoint_sha256 == evidence.checkpoint_sha256
    assert resumed.epochs_completed == evidence.epochs_completed
