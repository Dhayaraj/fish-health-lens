from pathlib import Path
import shutil

import kagglehub


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
DATASET = "irfanulhuda/fish-disease-detection-dataset"


def unique_destination(destination: Path) -> Path:
    if not destination.exists():
        return destination

    counter = 2
    while True:
        candidate = destination.with_name(
            f"{destination.stem}_{counter}{destination.suffix}"
        )
        if not candidate.exists():
            return candidate
        counter += 1


def copy_image(image_path: Path, class_name: str, output_root: Path) -> None:
    class_directory = output_root / class_name
    class_directory.mkdir(parents=True, exist_ok=True)
    destination = unique_destination(class_directory / image_path.name)
    shutil.copy2(image_path, destination)


def main() -> None:
    downloaded_path = Path(kagglehub.dataset_download(DATASET))
    dataset_root = downloaded_path / "New Dataset"
    train_root = dataset_root / "train_split"
    test_root = dataset_root / "test_split"
    classify_root = Path("data/classify")
    test_output_root = Path("data/test")

    if not train_root.is_dir() or not test_root.is_dir():
        raise RuntimeError(f"Expected train_split and test_split under {dataset_root}")

    train_images = 0
    class_names = []
    for class_directory in sorted(train_root.iterdir()):
        if not class_directory.is_dir():
            continue
        class_name = class_directory.name.strip()
        class_names.append(class_name)
        for image_path in class_directory.iterdir():
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                copy_image(image_path, class_name, classify_root)
                train_images += 1

    test_images = 0
    for image_path in test_root.iterdir():
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        matching_classes = [
            class_name
            for class_name in class_names
            if image_path.name.startswith(f"{class_name}_")
        ]
        if not matching_classes:
            print(f"Skipping test image with unknown class: {image_path.name}")
            continue
        copy_image(image_path, max(matching_classes, key=len), test_output_root)
        test_images += 1

    print(f"Downloaded dataset to: {downloaded_path}")
    print(f"Organized {train_images} training images under: {classify_root}")
    print(f"Organized {test_images} test images under: {test_output_root}")
    if train_images == 0 or test_images == 0:
        raise RuntimeError(
            "No images were found. Inspect the downloaded dataset layout before continuing."
        )


if __name__ == "__main__":
    main()