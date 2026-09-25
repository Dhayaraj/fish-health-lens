from pathlib import Path
import shutil

import torch
from ultralytics import YOLO


DATA_CONFIG = Path("data/detect/data.yaml")
OUTPUT_MODEL = Path("model/yolo_fish.pt")
BASE_MODEL = "yolo12n.pt"
EPOCHS = 50
IMAGE_SIZE = 640


def main() -> None:
    if not DATA_CONFIG.exists():
        raise FileNotFoundError(
            "Add YOLO annotations and data/detect/data.yaml before training."
        )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using Ultralytics device: {device}")
    model = YOLO(BASE_MODEL)
    results = model.train(
        data=str(DATA_CONFIG),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        device=device,
        project="runs/detect",
        name="fish",
        exist_ok=True,
    )

    best_model = Path(results.save_dir) / "weights/best.pt"
    if not best_model.exists():
        raise FileNotFoundError(f"Training completed without {best_model}.")
    OUTPUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_model, OUTPUT_MODEL)
    print(f"Saved detector to: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()