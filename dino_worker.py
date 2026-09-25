import base64
import io
import json
from pathlib import Path
import sys

import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_PATH = PROJECT_ROOT / "model/fish_disease_dino.pt"


def main() -> None:
    request = json.loads(sys.stdin.read())
    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    processor = AutoImageProcessor.from_pretrained(checkpoint["model_name"])
    backbone = AutoModel.from_pretrained(checkpoint["model_name"])
    backbone.eval()
    classifier = torch.nn.Sequential(
        torch.nn.Linear(checkpoint["hidden_size"], 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, len(checkpoint["class_names"])),
    )
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    classifier.eval()

    image = Image.open(io.BytesIO(base64.b64decode(request["image"]))).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        features = backbone(**inputs).last_hidden_state[:, 0, :]
        probabilities = torch.softmax(classifier(features), dim=1)[0].numpy()

    class_index = int(probabilities.argmax())
    print(json.dumps({
        "disease": checkpoint["class_names"][class_index],
        "confidence": float(probabilities[class_index]),
        "scores": {
            name: float(score)
            for name, score in zip(checkpoint["class_names"], probabilities)
        },
    }))


if __name__ == "__main__":
    main()
