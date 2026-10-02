"""Build a one-class COCO training set and a human label-review pack from AgRob."""

import argparse
import csv
import hashlib
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

from evaluate_agrob_fruit import (
    ARCHIVE,
    DATA_ROOT,
    EXPECTED_ARCHIVE_MD5,
    RIPENESS,
    load_images,
)


DEFAULT_OUTPUT = Path("data/agrob_finetune_v1")
DEFAULT_MANIFEST = Path("agrob_finetune_split.json")
VALIDATION_BLOCKS = ((70, 90), (225, 245))  # half-open positions in sorted pilot frames
BUFFER_FRAMES = 5


def assignment(pilot: list[dict], development: list[dict]) -> dict[str, str]:
    if len(pilot) != 297 or len(development) != 152:
        raise ValueError("AgRob image counts changed; inspect the source before reusing this split")
    result = {}
    for index, item in enumerate(pilot):
        if any(start <= index < end for start, end in VALIDATION_BLOCKS):
            split = "valid"
        elif any(start - BUFFER_FRAMES <= index < end + BUFFER_FRAMES for start, end in VALIDATION_BLOCKS):
            split = "buffer"
        else:
            split = "train"
        result[item["image"].name] = split
    for item in development:
        result[item["image"].name] = "development"
    return result


def read_override(path: Path, expected_name: str, expected_size: tuple[int, int]) -> list[dict]:
    root = ET.parse(path).getroot()
    if Path(root.findtext("filename", "")).name != expected_name:
        raise ValueError(f"Override filename disagrees with image: {path}")
    size = (int(root.findtext("size/width")), int(root.findtext("size/height")))
    if size != expected_size:
        raise ValueError(f"Override image size disagrees with source: {path}")
    result = []
    for obj in root.findall("object"):
        label = obj.findtext("name", "").strip().lower()
        if label not in RIPENESS | {"tomato"}:
            raise ValueError(f"Unexpected override label {label!r}: {path}")
        box = [float(obj.findtext(f"bndbox/{side}")) for side in ("xmin", "ymin", "xmax", "ymax")]
        if not (0 <= box[0] < box[2] <= size[0] and 0 <= box[1] < box[3] <= size[1]):
            raise ValueError(f"Invalid override box {box}: {path}")
        result.append({
            "label": label,
            "box": box,
            "occluded": obj.findtext("occluded", "0") == "1",
            "truncated": obj.findtext("truncated", "0") == "1",
            "difficult": obj.findtext("difficult", "0") == "1",
        })
    return result


def review_items(items: list[dict], splits: dict[str, str], count: int) -> list[dict]:
    """Select varied, repeatable examples; this does not certify the labels."""
    reasons: dict[str, set[str]] = defaultdict(set)

    def add(candidates: list[dict], name: str, limit: int) -> None:
        for item in candidates[:limit]:
            reasons[item["image"].name].add(name)

    add([item for item in items if not item["fruit"]][::5], "zero-fruit", 3)
    add(sorted(items, key=lambda item: len(item["fruit"]), reverse=True), "crowded", 8)
    add(sorted(items, key=lambda item: sum(f["occluded"] for f in item["fruit"]), reverse=True), "occluded", 8)
    add(sorted(items, key=lambda item: sum(f["label"] in {"reddish", "riped"} for f in item["fruit"]), reverse=True), "ripening", 8)
    old_counts = Path("results/agrob/test_any_counts.csv")
    if old_counts.is_file():
        with old_counts.open(newline="", encoding="utf-8") as source:
            worst = sorted(csv.DictReader(source), key=lambda row: int(row["absolute_count_error"]), reverse=True)[:8]
        for row in worst:
            reasons[row["file_name"]].add("baseline-count-error")
    rng = random.Random(42)
    for split in ("train", "valid", "development"):
        candidates = [item for item in items if splits[item["image"].name] == split]
        rng.shuffle(candidates)
        add(candidates, f"{split}-sample", 5)
    by_name = {item["image"].name: item for item in items}
    selected = [by_name[name] for name in sorted(reasons) if name in by_name]
    if len(selected) > count:
        selected = selected[:count]
    for item in selected:
        item["review_reasons"] = "; ".join(sorted(reasons[item["image"].name]))
    return selected


def make_review_pack(items: list[dict], splits: dict[str, str], output: Path, count: int) -> None:
    review_dir = output / "review"
    review_dir.mkdir()
    chosen = review_items(items, splits, count)
    with (review_dir / "review_queue.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=(
            "file_name", "split", "fruit_count", "ripeness_counts", "reason", "reviewed_by", "needs_correction", "notes"
        ))
        writer.writeheader()
        for item in chosen:
            writer.writerow({
                "file_name": item["image"].name,
                "split": splits[item["image"].name],
                "fruit_count": len(item["fruit"]),
                "ripeness_counts": json.dumps(dict(Counter(f["label"] for f in item["fruit"])), sort_keys=True),
                "reason": item["review_reasons"],
                "reviewed_by": "",
                "needs_correction": "",
                "notes": "",
            })
    thumb_width, thumb_height, caption_height = 384, 216, 40
    for page, start in enumerate(range(0, len(chosen), 10), 1):
        page_items = chosen[start:start + 10]
        rows = (len(page_items) + 1) // 2
        sheet = Image.new("RGB", (thumb_width * 2, (thumb_height + caption_height) * rows), "#101820")
        draw_sheet = ImageDraw.Draw(sheet)
        for offset, item in enumerate(page_items):
            with Image.open(item["image"]) as source:
                preview = source.convert("RGB").resize((thumb_width, thumb_height))
            draw = ImageDraw.Draw(preview)
            width, height = item["size"]
            for fruit in item["fruit"]:
                x1, y1, x2, y2 = fruit["box"]
                draw.rectangle((x1 * thumb_width / width, y1 * thumb_height / height,
                                x2 * thumb_width / width, y2 * thumb_height / height), outline="lime", width=2)
            x = (offset % 2) * thumb_width
            y = (offset // 2) * (thumb_height + caption_height)
            sheet.paste(preview, (x, y))
            draw_sheet.text((x + 5, y + thumb_height + 3), item["image"].stem[-34:], fill="white")
            draw_sheet.text((x + 5, y + thumb_height + 20),
                            f"{splits[item['image'].name]} | {len(item['fruit'])} fruit | {item['review_reasons'][:33]}", fill="#b7d9e8")
        sheet.save(review_dir / f"sheet_{page:02d}.jpg", quality=90)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--overrides", type=Path, help="Optional folder of corrected Pascal VOC XMLs; source files stay untouched")
    parser.add_argument("--review-record", type=Path, help="Completed review CSV; each row needs reviewer and yes/no correction decision")
    parser.add_argument("--review-count", type=int, default=50)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; use a new versioned directory")
    if args.review_count < 1:
        parser.error("review-count must be positive")
    if args.overrides and not args.overrides.is_dir():
        parser.error(f"Overrides directory is missing: {args.overrides}")
    if args.review_record and not args.review_record.is_file():
        parser.error(f"Review record is missing: {args.review_record}")
    archive_md5 = hashlib.md5(ARCHIVE.read_bytes()).hexdigest() if ARCHIVE.is_file() else None
    if archive_md5 != EXPECTED_ARCHIVE_MD5:
        raise ValueError("Source archive is missing or its MD5 does not match Zenodo")

    pilot = load_images(args.data_root, "pilot")
    development = load_images(args.data_root, "test")
    items = pilot + development
    splits = assignment(pilot, development)
    split_manifest = {
        "dataset": "AgRobTomato Zenodo 5596799",
        "archive_md5": archive_md5,
        "policy": "Two 20-frame validation blocks within the pilot sequence, five adjacent frames excluded on each side; other pilot frames train; previously viewed second sequence development only",
        "validation_positions_zero_based_half_open": [list(block) for block in VALIDATION_BLOCKS],
        "buffer_frames_each_side": BUFFER_FRAMES,
        "images": [{"file_name": name, "split": split} for name, split in sorted(splits.items())],
    }
    if args.split_manifest.exists():
        if json.loads(args.split_manifest.read_text(encoding="utf-8")) != split_manifest:
            raise ValueError(f"Saved split differs from source: {args.split_manifest}")
    else:
        args.split_manifest.write_bytes((json.dumps(split_manifest, indent=2) + "\n").encode("utf-8"))

    override_names = set()
    if args.overrides:
        for item in items:
            path = args.overrides / item["image"].with_suffix(".xml").name
            if path.is_file():
                item["fruit"] = read_override(path, item["image"].name, item["size"])
                override_names.add(path.name)
        unknown = {path.name for path in args.overrides.glob("*.xml")} - override_names
        if unknown:
            raise ValueError(f"Override XMLs have no AgRob image: {sorted(unknown)[:3]}")

    reviewed_rows = []
    if args.review_record:
        with args.review_record.open(newline="", encoding="utf-8-sig") as source:
            reader = csv.DictReader(source)
            if not {"file_name", "reviewed_by", "needs_correction"} <= set(reader.fieldnames or []):
                raise ValueError("Review record needs file_name, reviewed_by, and needs_correction columns")
            reviewed_rows = list(reader)
        if not reviewed_rows or len({row["file_name"] for row in reviewed_rows}) != len(reviewed_rows):
            raise ValueError("Review record must have unique image rows")
        for row in reviewed_rows:
            if row["file_name"] not in splits:
                raise ValueError(f"Unknown image in review record: {row['file_name']}")
            if not row["reviewed_by"].strip() or row["needs_correction"].strip().lower() not in {"yes", "no"}:
                raise ValueError(f"Incomplete review record: {row['file_name']}")
            if row["needs_correction"].strip().lower() == "yes" and Path(row["file_name"]).with_suffix(".xml").name not in override_names:
                raise ValueError(f"Correction was marked yes but XML override is missing: {row['file_name']}")

    args.output.mkdir(parents=True)
    annotations_dir = args.output / "annotations"
    annotations_dir.mkdir()
    summary = {"source_archive_md5": archive_md5, "split_manifest": str(args.split_manifest),
               "corrected_xml_files": len(override_names),
               "label_review_status": f"{len(reviewed_rows)} sampled images reviewed; other labels inherited from source"
               if reviewed_rows else "provisional; team review pending", "splits": {}}
    image_ids = {item["image"].name: index for index, item in enumerate(items, 1)}
    annotation_id = 1
    for split in ("train", "valid", "development"):
        selected = [item for item in items if splits[item["image"].name] == split]
        images, annotations = [], []
        counts = Counter()
        for item in selected:
            image_id = image_ids[item["image"].name]
            width, height = item["size"]
            images.append({"id": image_id, "file_name": item["image"].name, "width": width, "height": height})
            for fruit in item["fruit"]:
                x1, y1, x2, y2 = fruit["box"]
                bbox = [x1, y1, x2 - x1, y2 - y1]
                annotations.append({
                    "id": annotation_id, "image_id": image_id, "category_id": 0,
                    "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0,
                    "source_ripeness": fruit["label"], "occluded": fruit["occluded"],
                    "truncated": fruit["truncated"], "difficult": fruit["difficult"],
                })
                annotation_id += 1
                counts[fruit["label"]] += 1
        coco = {"info": {"description": "AgRob fruit counting; provisional one-class conversion"},
                "images": images, "annotations": annotations, "categories": [{"id": 0, "name": "tomato"}]}
        (annotations_dir / f"{split}.json").write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")
        summary["splits"][split] = {"images": len(selected), "fruit": len(annotations),
                                    "zero_fruit_images": sum(not item["fruit"] for item in selected),
                                    "source_ripeness": dict(sorted(counts.items()))}
    summary["splits"]["buffer"] = {"images": sum(split == "buffer" for split in splits.values())}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    make_review_pack(items, splits, args.output, args.review_count)
    if args.review_record:
        shutil.copyfile(args.review_record, args.output / "review" / "review_record.csv")
    print(json.dumps(summary, indent=2))
    print(f"Prepared data: {args.output.resolve()}")
    print(f"Review queue: {(args.output / 'review/review_queue.csv').resolve()}")


if __name__ == "__main__":
    main()
