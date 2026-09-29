"""Convert three CVAT COCO or CVAT XML exports into an RF-DETR-ready dataset."""

import argparse
import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path, PureWindowsPath


def cvat_xml_to_coco(xml_bytes: bytes) -> dict:
    root = ET.fromstring(xml_bytes)
    if root.tag != "annotations":
        raise ValueError(f"Expected CVAT <annotations> root, got <{root.tag}>")
    labels = [item.findtext("name") for item in root.findall("./meta/task/labels/label")]
    if labels != ["truss"]:
        raise ValueError(f"Expected one CVAT label named truss; got {labels}")
    coco = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "truss"}]}
    annotation_id = 1
    for image in root.findall("image"):
        image_id = int(image.attrib["id"])
        coco["images"].append({
            "id": image_id,
            "file_name": image.attrib["name"],
            "width": int(image.attrib["width"]),
            "height": int(image.attrib["height"]),
        })
        for shape in image:
            if shape.tag != "box" or shape.attrib.get("label") != "truss":
                raise ValueError(f"Unsupported shape or label in image {image.attrib['name']}: {shape.tag}")
            left = float(shape.attrib["xtl"])
            top = float(shape.attrib["ytl"])
            right = float(shape.attrib["xbr"])
            bottom = float(shape.attrib["ybr"])
            width = right - left
            height = bottom - top
            coco["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [left, top, width, height],
                "area": width * height,
                "iscrowd": 0,
            })
            annotation_id += 1
    return coco


def read_cvat_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        coco_candidates = [
            name for name in archive.namelist()
            if name.lower().endswith(".json") and "/annotations/" in f"/{name.lower()}"
        ]
        xml_candidates = [name for name in archive.namelist() if PureWindowsPath(name).name.lower() == "annotations.xml"]
        if len(coco_candidates) == 1 and not xml_candidates:
            return json.loads(archive.read(coco_candidates[0]))
        if len(xml_candidates) == 1 and not coco_candidates:
            return cvat_xml_to_coco(archive.read(xml_candidates[0]))
        raise ValueError(f"Expected one COCO JSON or CVAT annotations.xml in {path}; found {archive.namelist()}")


def convert(split: str, pilot_root: Path, export: Path, allow_empty: bool = False) -> tuple[int, int, int]:
    if not export.is_file():
        raise FileNotFoundError(f"Missing CVAT export: {export}")
    split_dir = pilot_root / split
    expected = {path.name for path in split_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    if not expected:
        raise ValueError(f"No pilot images found in {split_dir}")
    coco = read_cvat_zip(export)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])
    if len(categories) != 1 or categories[0].get("name", "").casefold() != "truss":
        raise ValueError(f"{split}: expected exactly one category named truss; got {categories}")
    by_name = {}
    image_by_id = {}
    for image in images:
        name = PureWindowsPath(str(image["file_name"])).name
        if name in by_name:
            raise ValueError(f"{split}: duplicate image name in COCO export: {name}")
        by_name[name] = image
        image_by_id[image["id"]] = image
        image["file_name"] = name
    missing = expected - set(by_name)
    extra = set(by_name) - expected
    if missing or extra:
        raise ValueError(f"{split}: image mismatch; missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}")

    for annotation in annotations:
        image = image_by_id.get(annotation.get("image_id"))
        if image is None or annotation.get("category_id") != categories[0]["id"]:
            raise ValueError(f"{split}: annotation has unknown image or category: {annotation.get('id')}")
        x, y, width, height = [float(value) for value in annotation["bbox"]]
        if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > image["width"] + 1 or y + height > image["height"] + 1:
            raise ValueError(f"{split}: invalid box in annotation {annotation.get('id')}")
    annotated_image_ids = {annotation["image_id"] for annotation in annotations}
    empty_names = [item["file_name"] for item in images if item["id"] not in annotated_image_ids]
    if empty_names and not allow_empty:
        raise ValueError(
            f"{split}: images without truss boxes: {empty_names}. "
            "Review them in CVAT and re-export; use --allow-empty only for confirmed negative images."
        )
    destination = split_dir / "_annotations.coco.json"
    destination.write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")
    return len(images), len(annotations), len(empty_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, default=Path("data/truss_pilot"))
    parser.add_argument("--exports", type=Path, default=Path("data/cvat_exports"))
    parser.add_argument("--allow-empty", action="store_true", help="Only after confirming images without boxes are genuine negatives")
    args = parser.parse_args()
    for split in ("train", "valid", "test"):
        try:
            count, boxes, no_boxes = convert(split, args.pilot_root, args.exports / f"{split}.zip", args.allow_empty)
        except (ValueError, FileNotFoundError, zipfile.BadZipFile, ET.ParseError) as error:
            parser.exit(1, f"Import stopped: {error}\n")
        print(f"{split}: {count} images, {boxes} truss boxes, {no_boxes} images without boxes", flush=True)
        if no_boxes:
            print(f"  Accepted {no_boxes} confirmed no-truss images in {split}.")
    print("COCO annotations are ready under each data/truss_pilot split folder.")


if __name__ == "__main__":
    main()
