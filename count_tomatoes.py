"""Count visible tomato detections in one image or a folder using a trained checkpoint."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from evaluate_agrob_fruit import save_preview
from finetune_agrob_fruit import detections_from_result, load_processor_and_model


DEFAULT_CHECKPOINT = Path("outputs/agrob-gpu-full-unpadded-v1-480/best_model")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def image_paths(path: Path) -> list[Path]:
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [path]
    if path.is_dir():
        files = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)
        if files:
            if len({item.stem.lower() for item in files}) != len(files):
                raise ValueError("Images share a base name, so previews would overwrite; rename one image")
            return files
    raise ValueError(f"No JPEG or PNG images found at {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True, help="One image or a folder of images")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--threshold", type=float, default=0.9, help="Chosen on AgRob validation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output", type=Path, required=True, help="A new folder for counts and previews")
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        parser.error("threshold must be between 0 and 1")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new folder")
    if not args.checkpoint.is_dir():
        parser.error(f"Checkpoint is missing: {args.checkpoint}")
    paths = image_paths(args.images)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but PyTorch cannot see a GPU")
    _, processor, model = load_processor_and_model(args.checkpoint, None)
    if processor.pad_size is not None:
        parser.error("Checkpoint uses invalid fixed padding; use the corrected unpadded checkpoint")
    model = model.to(device).eval()
    args.output.mkdir(parents=True)
    preview_dir = args.output / "previews"
    preview_dir.mkdir()
    rows = []
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
        with torch.no_grad():
            for index, path in enumerate(paths, 1):
                with Image.open(path) as source:
                    image = source.convert("RGB")
                encoded = processor(images=image, return_tensors="pt")
                output = model(pixel_values=encoded["pixel_values"].to(device),
                               pixel_mask=encoded["pixel_mask"].to(device))
                target_sizes = torch.tensor([[image.height, image.width]], device=device)
                result = processor.post_process_object_detection(output, threshold=args.threshold,
                                                                 target_sizes=target_sizes)[0]
                detections = detections_from_result(result, model)
                preview_path = preview_dir / (path.stem + ".jpg")
                save_preview(image, [], detections, preview_path)
                rows.append({"file_name": path.name, "visible_tomato_count": len(detections),
                             "preview": str(preview_path)})
                predictions.write(json.dumps({"file_name": path.name, "detections": detections}) + "\n")
                print(f"{index}/{len(paths)} {path.name}: {len(detections)} visible tomato detections", flush=True)
    with (args.output / "counts.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"checkpoint": str(args.checkpoint.resolve()), "threshold": args.threshold,
               "device": device, "images": len(rows), "run_utc": datetime.now(timezone.utc).isoformat(),
               "note": "These are model detections, not verified visible-fruit counts; inspect previews."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved counts and red-box previews in {args.output.resolve()}")


if __name__ == "__main__":
    main()
