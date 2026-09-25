from collections import Counter
from pathlib import Path

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
DATA_ROOTS = (Path("data/classify"), Path("data/test"))


def main() -> None:
    for data_root in DATA_ROOTS:
        if not data_root.exists():
            raise FileNotFoundError(
                f"{data_root} does not exist. Run download_data.py first."
            )

        counts = Counter()
        corrupt_files = []
        unsupported_files = []

        for file_path in sorted(data_root.rglob("*")):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                unsupported_files.append(file_path)
                continue

            class_name = file_path.parent.name
            try:
                with Image.open(file_path) as image:
                    image.verify()
                counts[class_name] += 1
            except (OSError, SyntaxError):
                corrupt_files.append(file_path)

        print(f"\n{data_root} images per class:")
        for class_name, count in sorted(counts.items()):
            print(f"  {class_name}: {count}")
        print(f"Total valid images: {sum(counts.values())}")
        print(f"Corrupt images: {len(corrupt_files)}")
        print(f"Unsupported files: {len(unsupported_files)}")

        if corrupt_files:
            print("\nCorrupt files:")
            for file_path in corrupt_files:
                print(f"  {file_path}")


if __name__ == "__main__":
    main()