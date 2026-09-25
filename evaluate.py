from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix


TEST_DIRECTORY = Path("data/test")
MODEL_PATH = Path("model/fish_disease.keras")
CLASSES_PATH = Path("model/classes.txt")
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


def main() -> None:
    if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
        raise FileNotFoundError("Train the model first with train.py.")
    if not TEST_DIRECTORY.exists():
        raise FileNotFoundError(
            f"{TEST_DIRECTORY} does not exist. Run download_data.py first."
        )

    class_names = CLASSES_PATH.read_text(encoding="utf-8").splitlines()
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIRECTORY,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        shuffle=False,
    )
    if test_dataset.class_names != class_names:
        raise ValueError(
            "The test directory classes do not match model/classes.txt: "
            f"{test_dataset.class_names} != {class_names}"
        )

    model = tf.keras.models.load_model(MODEL_PATH)
    probabilities = model.predict(test_dataset, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)
    true_labels = np.concatenate([labels.numpy() for _, labels in test_dataset])

    accuracy = np.mean(predicted_labels == true_labels)
    matrix = confusion_matrix(true_labels, predicted_labels)
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=range(len(class_names)),
        target_names=class_names,
        zero_division=0,
    )

    print(f"\nTest images: {len(true_labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(matrix)
    print("\nPer-class precision/recall:")
    print(report)


if __name__ == "__main__":
    main()