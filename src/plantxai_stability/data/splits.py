"""Leaf-safe deterministic split validation and construction."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Iterable

from plantxai_stability.contracts import SampleRecord


def validate_frozen_splits(records: Iterable[SampleRecord]) -> None:
    materialized = list(records)
    leaf_splits: dict[str, set[str]] = defaultdict(set)
    leaf_classes: dict[str, set[str]] = defaultdict(set)
    sample_ids: set[str] = set()
    for record in materialized:
        if record.sample_id in sample_ids:
            raise ValueError(f"Duplicate sample_id in split: {record.sample_id}")
        if record.split not in {"train", "validation", "test"}:
            raise ValueError(f"Unknown frozen split: {record.split}")
        if record.source_split == "test" and record.split != "test":
            raise ValueError(f"Official test sample moved out of test: {record.sample_id}")
        sample_ids.add(record.sample_id)
        leaf_splits[record.leaf_id].add(record.split)
        leaf_classes[record.leaf_id].add(record.class_name)
    leaked = [leaf for leaf, splits in leaf_splits.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"leaf_id appears in multiple splits: {leaked[:5]}")
    conflicting = [leaf for leaf, classes in leaf_classes.items() if len(classes) > 1]
    if conflicting:
        raise ValueError(f"leaf_id maps to multiple classes: {conflicting[:5]}")


def group_train_validation(records: Iterable[SampleRecord], validation_fraction: float, seed: int) -> list[SampleRecord]:
    """Split only records marked train; existing official test remains untouched."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    materialized = sorted(list(records), key=lambda item: item.sample_id)
    groups: dict[str, list[SampleRecord]] = defaultdict(list)
    for record in materialized:
        if record.source_split == "test":
            if record.split != "test":
                raise ValueError(f"Official test sample is not in test: {record.sample_id}")
            continue
        groups[record.leaf_id].append(record)
    leaves_by_class: dict[str, list[str]] = defaultdict(list)
    for leaf_id, items in groups.items():
        classes = {item.class_name for item in items}
        if len(classes) != 1:
            raise ValueError(f"leaf_id maps to multiple classes: {leaf_id}")
        leaves_by_class[next(iter(classes))].append(leaf_id)
    validation_leaves: set[str] = set()
    for class_name, class_leaves in sorted(leaves_by_class.items()):
        if len(class_leaves) < 2:
            raise ValueError(
                f"Cannot create leaf-safe stratified validation for class {class_name}: "
                "fewer than two training leaves"
            )
        shuffled = sorted(class_leaves)
        random.Random(f"{seed}:{class_name}").shuffle(shuffled)
        target = max(1, round(len(shuffled) * validation_fraction))
        target = min(target, len(shuffled) - 1)
        validation_leaves.update(shuffled[:target])
    result: list[SampleRecord] = []
    for record in materialized:
        if record.source_split == "test":
            result.append(record)
        else:
            split = "validation" if record.leaf_id in validation_leaves else "train"
            result.append(SampleRecord(**{**record.__dict__, "split": split}))
    validate_frozen_splits(result)
    return sorted(result, key=lambda item: item.sample_id)
