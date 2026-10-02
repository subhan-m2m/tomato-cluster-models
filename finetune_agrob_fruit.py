"""Fine-tune one-class fruit DETR and evaluate saved checkpoints on AgRob."""

import argparse
import csv
import json
import random
import statistics
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from evaluate_agrob_fruit import DATA_ROOT, MODEL_ID, MODEL_REVISION, match_boxes, save_preview


DEFAULT_PREPARED = Path("data/agrob_finetune_v1")
DEFAULT_IMAGES = DATA_ROOT / "JPEGImages"


def load_split(prepared: Path, split: str, images_dir: Path) -> list[dict]:
    path = prepared / "annotations" / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Prepared annotations missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("categories") != [{"id": 0, "name": "tomato"}]:
        raise ValueError(f"Expected one tomato category with ID 0: {path}")
    by_image = defaultdict(list)
    ids = set()
    for annotation in data["annotations"]:
        if annotation["id"] in ids:
            raise ValueError(f"Repeated annotation ID in {path}")
        ids.add(annotation["id"])
        if annotation["category_id"] != 0 or annotation.get("iscrowd", 0):
            raise ValueError(f"Unexpected category or crowd box in {path}")
        by_image[annotation["image_id"]].append(annotation)
    items = []
    image_ids = set()
    for image in data["images"]:
        if image["id"] in image_ids:
            raise ValueError(f"Repeated image ID in {path}")
        image_ids.add(image["id"])
        name = Path(image["file_name"])
        if name.name != str(name) or ".." in name.parts:
            raise ValueError(f"Unsafe image name in {path}: {name}")
        image_path = images_dir / name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        truth = []
        for annotation in by_image[image["id"]]:
            x, y, width, height = map(float, annotation["bbox"])
            if not (0 <= x < x + width <= image["width"] and 0 <= y < y + height <= image["height"]):
                raise ValueError(f"Invalid box in {path}: {annotation['id']}")
            truth.append({"box": [x, y, x + width, y + height],
                          "source_ripeness": annotation.get("source_ripeness", "unknown"),
                          "occluded": bool(annotation.get("occluded", False))})
        items.append({"id": image["id"], "path": image_path,
                      "size": (image["width"], image["height"]),
                      "annotations": by_image[image["id"]], "truth": truth})
    if set(by_image) - image_ids:
        raise ValueError(f"Annotations refer to absent images in {path}")
    if not items:
        raise ValueError(f"No images in {path}")
    return sorted(items, key=lambda item: item["path"].name)


def load_processor_and_model(checkpoint: str | Path | None, image_size: int | None):
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection

    source = checkpoint or MODEL_ID
    options = {} if checkpoint else {"revision": MODEL_REVISION}
    processor = AutoImageProcessor.from_pretrained(source, use_fast=False, **options)
    if image_size is not None:
        processor.size = {"max_height": image_size, "max_width": image_size}
        processor.pad_size = {"height": image_size, "width": image_size}
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="for .* copying from a non-meta parameter")
        model = AutoModelForObjectDetection.from_pretrained(
            source,
            id2label={0: "tomato"},
            label2id={"tomato": 0},
            ignore_mismatched_sizes=checkpoint is None,
            **options,
        )
    return torch, processor, model


def process_batch(items: list[dict], processor, device: str, with_labels: bool):
    images = []
    for item in items:
        with Image.open(item["path"]) as source:
            image = source.convert("RGB")
        if image.size != item["size"]:
            raise ValueError(f"Image dimensions disagree with annotations: {item['path']}")
        images.append(image)
    kwargs = {}
    if with_labels:
        kwargs["annotations"] = [
            {"image_id": item["id"], "annotations": [
                {key: annotation[key] for key in ("image_id", "category_id", "bbox", "area", "iscrowd")}
                for annotation in item["annotations"]
            ]}
            for item in items
        ]
    encoded = processor(images=images, return_tensors="pt", **kwargs)
    pixel_values = encoded["pixel_values"].to(device)
    pixel_mask = encoded.get("pixel_mask")
    if pixel_mask is not None:
        pixel_mask = pixel_mask.to(device)
    labels = None
    if with_labels:
        labels = [{key: value.to(device) for key, value in label.items()} for label in encoded["labels"]]
    return images, pixel_values, pixel_mask, labels


def batches(items: list[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def detections_from_result(result, model) -> list[dict]:
    return [
        {"label": model.config.id2label[int(label)], "score": float(score),
         "box": [float(value) for value in box.tolist()]}
        for box, score, label in zip(result["boxes"], result["scores"], result["labels"])
        if int(label) == 0
    ]


def validate_epoch(items: list[dict], model, processor, torch, device: str, batch_size: int) -> tuple[float, float]:
    model.eval()
    losses, errors = [], []
    with torch.no_grad():
        for group in batches(items, batch_size):
            _, pixels, mask, labels = process_batch(group, processor, device, with_labels=True)
            output = model(pixel_values=pixels, pixel_mask=mask, labels=labels)
            losses.append(float(output.loss))
            sizes = torch.tensor([[item["size"][1], item["size"][0]] for item in group], device=device)
            results = processor.post_process_object_detection(output, threshold=0.3, target_sizes=sizes)
            for item, result in zip(group, results):
                errors.append(abs(len(detections_from_result(result, model)) - len(item["truth"])))
    return statistics.mean(losses), statistics.mean(errors)


def train(args) -> None:
    import torch
    import transformers

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("Training requires a CUDA GPU. Use --allow-cpu only for a tiny plumbing check.")
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}; choose a new run folder")
    if args.epochs < 1 or args.batch_size < 1 or args.grad_accum < 1 or args.image_size < 64:
        raise SystemExit("epochs, batch size, and accumulation must be positive; image size must be at least 64")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    train_items = load_split(args.prepared, "train", args.images_dir)
    valid_items = load_split(args.prepared, "valid", args.images_dir)
    if args.max_train_images:
        train_items = train_items[:args.max_train_images]
    if args.max_valid_images:
        valid_items = valid_items[:args.max_valid_images]
    torch, processor, model = load_processor_and_model(None, args.image_size)
    model = model.to(device)
    backbone = list(model.model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone}
    optimizer = torch.optim.AdamW([
        {"params": backbone, "lr": args.backbone_lr},
        {"params": [p for p in model.parameters() if id(p) not in backbone_ids], "lr": args.learning_rate},
    ], weight_decay=1e-4)
    args.output.mkdir(parents=True)
    run_info = {
        "source_model": MODEL_ID, "source_revision": MODEL_REVISION,
        "prepared_summary": json.loads((args.prepared / "summary.json").read_text(encoding="utf-8")),
        "device": device, "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "seed": args.seed, "epochs": args.epochs, "batch_size": args.batch_size,
        "gradient_accumulation": args.grad_accum, "learning_rate": args.learning_rate,
        "backbone_learning_rate": args.backbone_lr, "image_size": args.image_size,
        "train_images": len(train_items), "valid_images": len(valid_items),
        "limited_run": bool(args.max_train_images or args.max_valid_images),
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    (args.output / "run_config.json").write_text(json.dumps(run_info, indent=2) + "\n", encoding="utf-8")
    best_loss = float("inf")
    history_path = args.output / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=("epoch", "train_loss", "valid_loss", "valid_count_mae_at_0_3", "best_epoch"))
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            shuffled = train_items.copy()
            random.Random(args.seed + epoch).shuffle(shuffled)
            groups = list(batches(shuffled, args.batch_size))
            losses = []
            optimizer.zero_grad(set_to_none=True)
            for step, group in enumerate(groups, 1):
                _, pixels, mask, labels = process_batch(group, processor, device, with_labels=True)
                output = model(pixel_values=pixels, pixel_mask=mask, labels=labels)
                if not torch.isfinite(output.loss):
                    raise RuntimeError(f"Non-finite training loss at epoch {epoch}, step {step}")
                losses.append(float(output.loss.detach()))
                remaining = len(groups) - ((step - 1) // args.grad_accum) * args.grad_accum
                divisor = min(args.grad_accum, remaining)
                (output.loss / divisor).backward()
                if step % args.grad_accum == 0 or step == len(groups):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                if step == 1 or step % 20 == 0 or step == len(groups):
                    print(f"epoch {epoch}/{args.epochs} batch {step}/{len(groups)} loss {losses[-1]:.4f}", flush=True)
            valid_loss, valid_mae = validate_epoch(valid_items, model, processor, torch, device, args.batch_size)
            improved = valid_loss < best_loss
            if improved:
                best_loss = valid_loss
                best_dir = args.output / "best_model"
                model.save_pretrained(best_dir, safe_serialization=True)
                processor.save_pretrained(best_dir)
                (args.output / "best_epoch.json").write_text(
                    json.dumps({"epoch": epoch, "valid_loss": valid_loss, "valid_count_mae_at_0_3": valid_mae}, indent=2) + "\n",
                    encoding="utf-8",
                )
            writer.writerow({"epoch": epoch, "train_loss": round(statistics.mean(losses), 5),
                             "valid_loss": round(valid_loss, 5), "valid_count_mae_at_0_3": round(valid_mae, 3),
                             "best_epoch": improved})
            history_file.flush()
            print(f"epoch {epoch}: train loss {statistics.mean(losses):.4f}, validation loss {valid_loss:.4f}, "
                  f"count MAE at 0.30 {valid_mae:.3f}" + (" (saved best)" if improved else ""), flush=True)
    print(f"Saved best checkpoint in {(args.output / 'best_model').resolve()}", flush=True)


def evaluate(args) -> None:
    import torch

    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}; choose a new run folder")
    thresholds = [float(value) for value in args.thresholds.split(",")]
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise SystemExit("Thresholds must be comma-separated numbers between 0 and 1")
    if args.split != "valid" and len(thresholds) != 1:
        raise SystemExit("Only validation may sweep thresholds; pass one frozen threshold for other splits")
    if not Path(args.checkpoint).is_dir():
        raise SystemExit(f"Model checkpoint is missing: {args.checkpoint}")
    device = "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    items = load_split(args.prepared, args.split, args.images_dir)
    if args.limit:
        items = items[:args.limit]
    torch, processor, model = load_processor_and_model(args.checkpoint, None)
    model = model.to(device).eval()
    raw = []
    with torch.no_grad():
        for index, group in enumerate(batches(items, args.batch_size), 1):
            _, pixels, mask, _ = process_batch(group, processor, device, with_labels=False)
            output = model(pixel_values=pixels, pixel_mask=mask)
            sizes = torch.tensor([[item["size"][1], item["size"][0]] for item in group], device=device)
            results = processor.post_process_object_detection(output, threshold=0.0, target_sizes=sizes)
            raw.extend(detections_from_result(result, model) for result in results)
            print(f"evaluated {min(index * args.batch_size, len(items))}/{len(items)}", flush=True)
    metrics = []
    all_rows = {}
    all_missed = {}
    for threshold in thresholds:
        rows = []
        missed_by_ripeness = Counter()
        total_by_ripeness = Counter()
        for item, detections in zip(items, raw):
            selected = [d for d in detections if d["score"] >= threshold]
            matches, missed, extra = match_boxes(item["truth"], selected)
            for fruit in item["truth"]:
                total_by_ripeness[fruit["source_ripeness"]] += 1
            for index in missed:
                missed_by_ripeness[item["truth"][index]["source_ripeness"]] += 1
            signed = len(selected) - len(item["truth"])
            rows.append({"file_name": item["path"].name, "ground_truth_fruit_count": len(item["truth"]),
                         "predicted_fruit_count": len(selected), "signed_count_error": signed,
                         "absolute_count_error": abs(signed), "matched_fruit_iou_50": len(matches),
                         "missed_fruit_iou_50": len(missed), "extra_boxes_iou_50": len(extra)})
        matched = sum(row["matched_fruit_iou_50"] for row in rows)
        missed_count = sum(row["missed_fruit_iou_50"] for row in rows)
        extra_count = sum(row["extra_boxes_iou_50"] for row in rows)
        precision = matched / (matched + extra_count) if matched + extra_count else 0.0
        recall = matched / (matched + missed_count) if matched + missed_count else 0.0
        summary = {"threshold": threshold, "images": len(items),
                   "ground_truth_fruit": sum(row["ground_truth_fruit_count"] for row in rows),
                   "predicted_fruit": sum(row["predicted_fruit_count"] for row in rows),
                   "mean_absolute_count_error": round(statistics.mean(row["absolute_count_error"] for row in rows), 3),
                   "mean_signed_count_error": round(statistics.mean(row["signed_count_error"] for row in rows), 3),
                   "exact_count_images": sum(row["absolute_count_error"] == 0 for row in rows),
                   "matched_fruit_iou_50": matched, "missed_fruit_iou_50": missed_count,
                   "extra_boxes_iou_50": extra_count, "box_precision_iou_50": round(precision, 4),
                   "box_recall_iou_50": round(recall, 4),
                   "box_f1_iou_50": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0}
        metrics.append(summary)
        all_rows[threshold] = rows
        all_missed[threshold] = {label: {"missed": missed_by_ripeness[label], "total": total_by_ripeness[label]}
                                 for label in sorted(total_by_ripeness)}
    chosen = min(metrics, key=lambda metric: (metric["mean_absolute_count_error"], -metric["box_f1_iou_50"], metric["threshold"]))
    threshold = chosen["threshold"]
    args.output.mkdir(parents=True)
    with (args.output / "thresholds.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    with (args.output / "counts.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=all_rows[threshold][0].keys())
        writer.writeheader()
        writer.writerows(all_rows[threshold])
    preview_dir = args.output / "previews"
    preview_dir.mkdir()
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as destination:
        for item, detections in zip(items, raw):
            selected = [d for d in detections if d["score"] >= threshold]
            with Image.open(item["path"]) as source:
                image = source.convert("RGB")
            save_preview(image, item["truth"], selected, preview_dir / item["path"].name)
            destination.write(json.dumps({"file_name": item["path"].name,
                                          "ground_truth": item["truth"], "detections": selected}) + "\n")
    report = {"checkpoint": str(Path(args.checkpoint).resolve()), "split": args.split,
              "run_utc": datetime.now(timezone.utc).isoformat(), "device": device,
              "limited_run": bool(args.limit), "chosen_by": "lowest validation count MAE, then highest box F1"
              if args.split == "valid" else "fixed threshold supplied before scoring",
              "chosen_metrics": chosen, "missed_by_ripeness": all_missed[threshold],
              "note": "The development split was inspected during the original baseline and is not a fresh final test."}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["chosen_metrics"], indent=2), flush=True)
    print(f"Saved evaluation in {args.output.resolve()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    train_parser = subcommands.add_parser("train", help="Fine-tune the one-class tomato detector")
    train_parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    train_parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    train_parser.add_argument("--output", type=Path, required=True)
    train_parser.add_argument("--epochs", type=int, default=15)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--grad-accum", type=int, default=2)
    train_parser.add_argument("--learning-rate", type=float, default=5e-5)
    train_parser.add_argument("--backbone-lr", type=float, default=1e-5)
    train_parser.add_argument("--image-size", type=int, default=640)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--allow-cpu", action="store_true", help="For a tiny plumbing check only")
    train_parser.add_argument("--cpu-threads", type=int, default=4)
    train_parser.add_argument("--max-train-images", type=int, default=0)
    train_parser.add_argument("--max-valid-images", type=int, default=0)
    train_parser.set_defaults(func=train)
    eval_parser = subcommands.add_parser("evaluate", help="Score a saved checkpoint")
    eval_parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    eval_parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--split", default="valid", help="Prepared split name; development is a viewed benchmark")
    eval_parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5")
    eval_parser.add_argument("--batch-size", type=int, default=2)
    eval_parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    eval_parser.add_argument("--limit", type=int, default=0)
    eval_parser.add_argument("--output", type=Path, required=True)
    eval_parser.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
