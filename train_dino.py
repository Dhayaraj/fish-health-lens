from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoImageProcessor, AutoModel


DATA_DIRECTORY = Path("data/classify")
MODEL_DIRECTORY = Path("model")
MODEL_PATH = MODEL_DIRECTORY / "fish_disease_dino.pt"
MODEL_NAME = "facebook/dinov2-base"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EPOCHS = 15
BATCH_SIZE = 16
SEED = 42


class FishDataset(Dataset):
    def __init__(self, root: Path, class_names: list[str]) -> None:
        self.samples = [
            (image_path, class_index)
            for class_index, class_name in enumerate(class_names)
            for image_path in sorted((root / class_name).iterdir())
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Image.Image, int]:
        image_path, label = self.samples[index]
        return Image.open(image_path).convert("RGB"), label


def make_collate(processor):
    def collate(batch):
        images, labels = zip(*batch)
        inputs = processor(images=list(images), return_tensors="pt")
        return inputs, torch.tensor(labels, dtype=torch.long)

    return collate


def features(model, inputs, device):
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        return model(**inputs).last_hidden_state[:, 0, :]


def main() -> None:
    if not DATA_DIRECTORY.exists():
        raise FileNotFoundError("Run download_data.py before train_dino.py.")

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    class_names = sorted(path.name for path in DATA_DIRECTORY.iterdir() if path.is_dir())
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    backbone = AutoModel.from_pretrained(MODEL_NAME).to(device)
    backbone.eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    dataset = FishDataset(DATA_DIRECTORY, class_names)
    validation_size = max(1, int(len(dataset) * 0.2))
    train_dataset, validation_dataset = random_split(
        dataset, [len(dataset) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(SEED),
    )
    collate = make_collate(processor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
    validation_loader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    classifier = nn.Sequential(
        nn.Linear(backbone.config.hidden_size, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(class_names)),
    ).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    loss_function = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        classifier.train()
        correct = total = running_loss = 0.0
        for inputs, labels in train_loader:
            labels = labels.to(device)
            logits = classifier(features(backbone, inputs, device))
            loss = loss_function(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)

        classifier.eval()
        validation_correct = validation_total = 0
        with torch.no_grad():
            for inputs, labels in validation_loader:
                labels = labels.to(device)
                logits = classifier(features(backbone, inputs, device))
                validation_correct += (logits.argmax(1) == labels).sum().item()
                validation_total += len(labels)
        print(
            f"Epoch {epoch + 1}/{EPOCHS} - loss {running_loss / total:.4f} - "
            f"train {correct / total:.2%} - validation {validation_correct / validation_total:.2%}"
        )

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": MODEL_NAME,
            "class_names": class_names,
            "hidden_size": backbone.config.hidden_size,
            "classifier_state_dict": classifier.state_dict(),
        },
        MODEL_PATH,
    )
    print(f"Saved DINOv2 classifier to: {MODEL_PATH}")


if __name__ == "__main__":
    main()