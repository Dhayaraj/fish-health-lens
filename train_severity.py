from pathlib import Path
import json

import cv2
import numpy as np
import xgboost as xgb


DATA_DIRECTORY = Path("data/classify")
MODEL_PATH = Path("model/severity.json")
FEATURE_NAMES = ["pct_affected", "num_fragments", "largest_fragment_area", "mean_hue", "mean_saturation", "mean_value", "body_area"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def extract_features(image: np.ndarray) -> dict[str, float]:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    body_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    white = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
    red_low = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    red_high = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    lesion_mask = cv2.bitwise_or(white, cv2.bitwise_or(red_low, red_high))
    lesion_mask = cv2.bitwise_and(lesion_mask, lesion_mask, mask=body_mask)
    contours, _ = cv2.findContours(lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    body_area = float(np.count_nonzero(body_mask))
    lesion_area = float(np.count_nonzero(lesion_mask))
    pixels = hsv[body_mask.astype(bool)]
    return {
        "pct_affected": lesion_area / body_area * 100,
        "num_fragments": float(len(contours)),
        "largest_fragment_area": float(max((cv2.contourArea(c) for c in contours), default=0)),
        "mean_hue": float(pixels[:, 0].mean()),
        "mean_saturation": float(pixels[:, 1].mean()),
        "mean_value": float(pixels[:, 2].mean()),
        "body_area": body_area,
    }


def label_from_percentage(value: float) -> int:
    return 0 if value < 5 else 1 if value < 20 else 2


def main() -> None:
    rows, labels = [], []
    for class_directory in sorted(DATA_DIRECTORY.iterdir()):
        if not class_directory.is_dir():
            continue
        for image_path in sorted(class_directory.iterdir())[:150]:
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            values = extract_features(image)
            rows.append([values[name] for name in FEATURE_NAMES])
            labels.append(label_from_percentage(values["pct_affected"]))

    model = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, eval_metric="mlogloss")
    model.fit(np.asarray(rows), np.asarray(labels))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    MODEL_PATH.with_suffix(".features.json").write_text(json.dumps(FEATURE_NAMES), encoding="utf-8")
    print(f"Saved experimental severity model to: {MODEL_PATH} ({len(rows)} images)")


if __name__ == "__main__":
    main()