"""Convert a reviewed Pascal VOC tomato image folder into one-class COCO JSON."""

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image

from evaluate_agrob_fruit import RIPENESS


def convert(source: Path) -> tuple[dict, dict]:
    image_dir = source / "JPEGImages"
    annotation_dir = source / "Annotations"
    if not image_dir.is_dir() or not annotation_dir.is_dir():
        raise FileNotFoundError(f"Expected JPEGImages/ and Annotations/ under {source}")
    images, annotations = [], []
    label_counts = Counter()
    image_files = sorted(image_dir.glob("*.jpg"))
    xml_files = {path.stem: path for path in annotation_dir.glob("*.xml")}
    if not image_files or len(image_files) != len(xml_files):
        raise ValueError("Every image must have exactly one XML annotation, including confirmed zero-fruit images")
    annotation_id = 1
    for image_id, image_path in enumerate(image_files, 1):
        xml_path = xml_files.pop(image_path.stem, None)
        if xml_path is None:
            raise FileNotFoundError(f"Missing XML for {image_path.name}")
        root = ET.parse(xml_path).getroot()
        if Path(root.findtext("filename", "")).name != image_path.name:
            raise ValueError(f"XML filename does not match {image_path.name}")
        with Image.open(image_path) as image:
            width, height = image.size
        if (int(root.findtext("size/width")), int(root.findtext("size/height"))) != (width, height):
            raise ValueError(f"XML image size does not match {image_path.name}")
        images.append({"id": image_id, "file_name": image_path.name, "width": width, "height": height})
        for obj in root.findall("object"):
            label = obj.findtext("name", "").strip().lower()
            if label not in RIPENESS | {"tomato"}:
                raise ValueError(f"Unexpected label {label!r} in {xml_path}")
            x1, y1, x2, y2 = [float(obj.findtext(f"bndbox/{side}"))
                               for side in ("xmin", "ymin", "xmax", "ymax")]
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                raise ValueError(f"Invalid box in {xml_path}")
            box = [x1, y1, x2 - x1, y2 - y1]
            annotations.append({"id": annotation_id, "image_id": image_id, "category_id": 0,
                                "bbox": box, "area": box[2] * box[3], "iscrowd": 0,
                                "source_ripeness": label,
                                "occluded": obj.findtext("occluded", "0") == "1"})
            annotation_id += 1
            label_counts[label] += 1
    if xml_files:
        raise ValueError(f"XML files have no matching image: {sorted(xml_files)[:3]}")
    coco = {"info": {"description": "Reviewed tomato count evaluation images"},
            "images": images, "annotations": annotations, "categories": [{"id": 0, "name": "tomato"}]}
    summary = {"images": len(images), "fruit": len(annotations),
               "zero_fruit_images": len({image["id"] for image in images} - {a["image_id"] for a in annotations}),
               "source_labels": dict(sorted(label_counts.items()))}
    return coco, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Folder with JPEGImages/ and Annotations/")
    parser.add_argument("--output", type=Path, required=True, help="New COCO JSON path")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    coco, summary = convert(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
