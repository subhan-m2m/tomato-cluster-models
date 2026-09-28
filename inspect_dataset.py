"""Check that the extracted Laboro COCO test split is ready to use."""

import argparse
import json
from collections import Counter
from pathlib import Path


def image_path(root: Path, split: str, file_name: str) -> Path | None:
    name = Path(file_name.replace("\\", "/"))
    if name.is_absolute() or ".." in name.parts:
        raise ValueError(f"Unsafe image path in annotations: {file_name}")
    for candidate in (root / split / name, root / name):
        if candidate.is_file():
            return candidate
    return None


def load_split(root: Path, split: str) -> tuple[dict, list[tuple[dict, Path]]]:
    annotation_file = root / "annotations" / f"{split}.json"
    if not annotation_file.is_file():
        raise FileNotFoundError(
            f"Missing {annotation_file}. Extract the tomato_mixed dataset so "
            "annotations/test.json and test/ are under data/laboro_tomato."
        )
    data = json.loads(annotation_file.read_text(encoding="utf-8"))
    images = data.get("images", [])
    annotations = data.get("annotations", [])
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError("COCO JSON must contain images and annotations lists")
    found = []
    missing = []
    for item in images:
        path = image_path(root, split, item["file_name"])
        if path is None:
            missing.append(item["file_name"])
        else:
            found.append((item, path))
    if missing:
        sample = ", ".join(missing[:3])
        raise FileNotFoundError(f"{len(missing)} annotated images missing; examples: {sample}")
    return data, found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/laboro_tomato"))
    parser.add_argument("--split", choices=("train", "test"), default="test")
    args = parser.parse_args()
    data, found = load_split(args.data_root, args.split)
    categories = {item["id"]: item["name"] for item in data.get("categories", [])}
    counts = Counter(categories.get(item["category_id"], "unknown") for item in data["annotations"])
    print(f"Split: {args.split}")
    print(f"Images found: {len(found)}")
    print(f"Fruit annotations: {len(data['annotations'])}")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")
    print("Truss/cluster labels: unavailable in Laboro Tomato")


if __name__ == "__main__":
    main()
