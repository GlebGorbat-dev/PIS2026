from __future__ import annotations

import torch
from PIL import Image

from src.model.architecture import build_model
from src.model.dataset import LeafCsvDataset, build_transforms, open_rgb
from src.model.predict import load_checkpoint, predict
from src.paths import load_train_config


def test_open_rgb_drops_alpha_channel(tmp_path):
    path = tmp_path / "rgba.png"
    Image.new("RGBA", (80, 80), (10, 20, 30, 40)).save(path)
    image = open_rgb(path)
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (10, 20, 30)


def test_csv_dataset_returns_tensor_and_label(tmp_path):
    image_path = tmp_path / "leaf.png"
    Image.new("RGBA", (120, 90), (40, 80, 20, 255)).save(image_path)
    csv_path = tmp_path / "split.csv"
    csv_path.write_text(
        f"path,class_name\n{image_path.name},healthy\n",
        encoding="utf-8",
    )
    config = load_train_config()
    dataset = LeafCsvDataset(
        csv_path,
        tmp_path,
        {"healthy": 0, "yellow_rust": 1},
        build_transforms(config, train=False),
    )
    tensor, label = dataset[0]
    assert tensor.shape[0] == 3
    assert tensor.shape[-1] == config["image_size"]
    assert label == 0


def test_predict_returns_class_and_confidence(tmp_path):
    names = ["healthy", "yellow_rust", "brown_rust", "septoria", "mildew"]
    model = build_model(len(names), pretrained=False, freeze_backbone=False)
    checkpoint = tmp_path / "toy.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "class_names": names,
            "config": load_train_config(),
            "best_val_accuracy": 0.0,
            "best_epoch": 1,
            "history": [],
            "architecture": "mobilenet_v3_small",
        },
        checkpoint,
    )
    import src.model.predict as predict_module

    predict_module._CACHE = None
    image = Image.new("RGB", (224, 224), (30, 90, 40))
    result = predict(image, weights_path=checkpoint)
    assert result.class_name in names
    assert 0.0 <= result.confidence <= 1.0
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-5
    assert set(result.probabilities) == set(names)
    predict_module._CACHE = None
