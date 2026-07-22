from PIL import Image

from plantxai_stability.data.huggingface import build_hf_manifest, inspect_hf_schema


class _Label:
    names = ["Tomato___healthy", "Tomato___Early_blight"]


class _Split(list):
    features = {"image": object(), "label": _Label(), "leaf_id": object()}


def test_hf_schema_and_manifest_preserve_leaf_and_split(tmp_path):
    dataset = {
        "train": _Split([
            {"image": Image.new("RGB", (4, 4), (10, 20, 30)), "label": 0, "leaf_id": "leaf-a"}
        ]),
        "test": _Split([
            {"image": Image.new("RGB", (4, 4), (30, 20, 10)), "label": 1, "leaf_id": "leaf-b"}
        ]),
    }
    schema = inspect_hf_schema(dataset)
    records = build_hf_manifest(dataset, schema, materialize_root=tmp_path)
    assert schema.has_leaf_id
    assert schema.splits == ("train", "test")
    assert {record.split for record in records} == {"train", "test"}
    assert {record.leaf_id for record in records} == {"leaf-a", "leaf-b"}
    assert (tmp_path / records[0].canonical_relative_path).is_file()


def test_materialization_preserves_distinct_sources_with_identical_pixels(tmp_path):
    source_a = tmp_path / "source-a.jpg"
    source_b = tmp_path / "source-b.jpg"
    image = Image.new("RGB", (4, 4), (10, 20, 30))
    image.save(source_a)
    image.save(source_b)
    dataset = {
        "train": _Split(
            [
                {
                    "image": {"path": str(source_a), "bytes": None},
                    "image_path": "raw/color/Tomato___healthy/source-a.jpg",
                    "label": 0,
                    "leaf_id": "leaf-a",
                },
                {
                    "image": {"path": str(source_b), "bytes": None},
                    "image_path": "raw/color/Tomato___healthy/source-b.jpg",
                    "label": 0,
                    "leaf_id": "leaf-b",
                },
            ]
        )
    }
    schema = inspect_hf_schema(dataset)
    records = build_hf_manifest(dataset, schema, materialize_root=tmp_path / "out")
    assert len(records) == 2
    assert len({record.sample_id for record in records}) == 2
    assert len({record.canonical_relative_path for record in records}) == 2
    assert len({record.canonical_rgb_sha256 for record in records}) == 1
