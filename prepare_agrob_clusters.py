"""Validate a supplied COCO cluster ZIP and prepare a fixed one-class dataset."""

import argparse
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image, ImageDraw


CLASS_NAME = "tomato_cluster"
CLUSTER_DEFINITION = "Visually grouped nearby tomatoes that appear to share one stem; stem membership is unverified."


def prepare(archive: Path, output: Path, manifest_path: Path) -> dict:
    if output.exists() or manifest_path.exists():
        raise ValueError("Choose new dataset and manifest paths; earlier versions are preserved")
    archive_hash = hashlib.sha256(archive.read_bytes()).hexdigest()
    payloads = {}
    manifest = {"source_archive": archive.name, "source_sha256": archive_hash,
                "source_label": "tomato cluster", "model_class": CLASS_NAME,
                "cluster_definition": CLUSTER_DEFINITION,
                "split_policy": "preserve supplied train/valid/test assignments", "splits": {}}
    originals = {}
    content_hashes = {}
    frames = []
    seen_names = set()
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.namelist()
        if len(entries) != len(set(entries)):
            raise ValueError("Repeated ZIP entry names")
        for name in entries:
            p = PurePosixPath(name)
            if p.is_absolute() or ".." in p.parts or "\\" in name or ":" in name:
                raise ValueError(f"Unsafe archive entry: {name}")
        for split in ("train", "valid", "test"):
            data = json.loads(bundle.read(f"{split}/_annotations.coco.json"))
            categories = {c["id"]: c["name"] for c in data["categories"]}
            cluster_ids = {i for i, label in categories.items() if label.casefold() == "tomato cluster"}
            if len(cluster_ids) != 1:
                raise ValueError(f"Expected one tomato cluster label: {split} {categories}")
            source_ids = {i["id"] for i in data["images"]}
            if len(source_ids) != len(data["images"]):
                raise ValueError(f"Repeated image IDs: {split}")
            image_lookup = {i["id"]: i for i in data["images"]}
            normalized_annotations = []
            annotation_ids = set()
            for annotation in data["annotations"]:
                if annotation["id"] in annotation_ids:
                    raise ValueError(f"Repeated annotation ID: {split}")
                annotation_ids.add(annotation["id"])
                if annotation["image_id"] not in source_ids or annotation["category_id"] not in cluster_ids:
                    raise ValueError(f"Unexpected annotation target or category: {split} {annotation['id']}")
                if annotation.get("iscrowd", 0):
                    raise ValueError("Cluster counting requires individual boxes, not crowd regions")
                image = image_lookup[annotation["image_id"]]
                box = list(map(float, annotation["bbox"]))
                if len(box) != 4 or not all(math.isfinite(v) for v in box):
                    raise ValueError(f"Invalid box: {split} {annotation['id']}")
                x, y, width, height = box
                if not (0 <= x < x + width <= image["width"] and 0 <= y < y + height <= image["height"]):
                    raise ValueError(f"Box outside image: {split} {annotation['id']} {box}")
                normalized_annotations.append({"id": annotation["id"], "image_id": annotation["image_id"],
                    "category_id": 0, "bbox": box, "area": width * height, "iscrowd": 0,
                    "source_label": "tomato cluster"})
            records = []
            for image in data["images"]:
                filename = image["file_name"]
                if PurePosixPath(filename).name != filename or filename in seen_names:
                    raise ValueError(f"Unsafe or repeated image filename: {filename}")
                seen_names.add(filename)
                original = image.get("extra", {}).get("name", filename)
                if original in originals:
                    raise ValueError(f"Source image occurs multiple times: {original}")
                originals[original] = split
                content = bundle.read(f"{split}/{filename}")
                fingerprint = hashlib.sha256(content).hexdigest()
                if fingerprint in content_hashes and content_hashes[fingerprint] != split:
                    raise ValueError("Identical image bytes occur across splits")
                content_hashes[fingerprint] = split
                with Image.open(io.BytesIO(content)) as source:
                    if source.size != (image["width"], image["height"]):
                        raise ValueError(f"Dimensions disagree with annotations: {filename}")
                    source.verify()
                payloads[filename] = content
                records.append({"id": image["id"], "file_name": filename, "original_name": original,
                                "width": image["width"], "height": image["height"], "sha256": fingerprint})
                match = re.match(r"(.+)_([0-9]+)\.[^.]+$", original)
                if match:
                    frames.append((match[1], int(match[2]), split, original))
            manifest["splits"][split] = records
            normalized = {"info": {"source_sha256": archive_hash}, "licenses": data.get("licenses", []),
                          "categories": [{"id": 0, "name": CLASS_NAME}],
                          "images": data["images"], "annotations": normalized_annotations}
            payloads[f"annotations/{split}.json"] = (json.dumps(normalized, indent=2) + "\n").encode()
    nearby = []
    for index, first in enumerate(frames):
        for second in frames[index + 1:]:
            if first[0] == second[0] and first[2] != second[2] and abs(first[1] - second[1]) <= 2:
                nearby.append({"first": first[3], "first_split": first[2], "second": second[3],
                               "second_split": second[2], "frame_gap": abs(first[1] - second[1])})
    manifest["nearby_frames_across_splits"] = nearby
    manifest["limitations"] = ["No confirmed zero-cluster images in this export",
        "AgRob video frames may overlap; nearby frames cross supplied splits",
        "Clusters are visual groups that appear to share a stem; botanical membership is unverified"]
    output.mkdir(parents=True)
    (output / "images").mkdir()
    (output / "annotations").mkdir()
    summary = {"source_archive_sha256": archive_hash, "class_name": CLASS_NAME,
               "cluster_definition": CLUSTER_DEFINITION,
               "split_manifest": str(manifest_path), "nearby_frame_pairs_across_splits": len(nearby),
               "splits": {}}
    for name, content in payloads.items():
        target = output / name if name.startswith("annotations/") else output / "images" / name
        target.write_bytes(content)
    for split in ("train", "valid", "test"):
        data = json.loads((output / "annotations" / f"{split}.json").read_text())
        counts = Counter(a["image_id"] for a in data["annotations"])
        summary["splits"][split] = {"images": len(data["images"]), "clusters": len(data["annotations"]),
                                    "zero_cluster_images": sum(counts[i["id"]] == 0 for i in data["images"])}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    # View training and validation labels before any model/test review.
    for split in ("train", "valid"):
        data = json.loads((output / "annotations" / f"{split}.json").read_text())
        tiles = []
        for item in sorted(data["images"], key=lambda i: i["file_name"])[:6]:
            with Image.open(output / "images" / item["file_name"]) as source:
                preview = source.convert("RGB")
            draw = ImageDraw.Draw(preview)
            for annotation in data["annotations"]:
                if annotation["image_id"] == item["id"]:
                    x, y, w, h = annotation["bbox"]
                    draw.rectangle((x, y, x + w, y + h), outline="lime", width=4)
            preview.thumbnail((640, 360))
            tiles.append(preview)
        sheet = Image.new("RGB", (1280, 360 * 3), "white")
        for index, tile in enumerate(tiles):
            sheet.paste(tile, ((index % 2) * 640, (index // 2) * 360))
        sheet.save(output / f"{split}_label_preview.jpg", quality=90)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/agrob_clusters_v1"))
    parser.add_argument("--manifest", type=Path, default=Path("agrob_cluster_manifest.json"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.archive, args.output, args.manifest), indent=2))
