"""Count annotated visual cluster detections in new images with the trained DETR."""

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from count_tomatoes import image_paths
from evaluate_agrob_fruit import save_preview
from finetune_agrob_fruit import detections_from_result, load_processor_and_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/agrob-cluster-detr-v2-480/best_model"))
    parser.add_argument("--threshold", type=float, required=True, help="Frozen threshold selected on validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1 or args.output.exists():
        parser.error("Use a threshold in 0–1 and a new output folder")
    if not args.checkpoint.is_dir():
        parser.error(f"Checkpoint is missing: {args.checkpoint}")
    paths = image_paths(args.images)
    torch, processor, model = load_processor_and_model(args.checkpoint, None, "tomato_cluster")
    if processor.pad_size is not None:
        raise ValueError("Use a checkpoint trained with dynamic padding")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    previews = args.output / "previews"
    previews.mkdir(parents=True)
    rows = []
    with (args.output / "predictions.jsonl").open("w", encoding="utf-8") as destination:
        with torch.no_grad():
            for index, path in enumerate(paths, 1):
                with Image.open(path) as source:
                    image = source.convert("RGB")
                encoded = processor(images=image, return_tensors="pt").to(device)
                output = model(**encoded)
                sizes = torch.tensor([[image.height, image.width]], device=device)
                result = processor.post_process_object_detection(output, threshold=args.threshold, target_sizes=sizes)[0]
                detections = detections_from_result(result, model)
                preview = previews / (path.stem + ".jpg")
                save_preview(image, [], detections, preview)
                row = {"file_name": path.name, "predicted_cluster_count": len(detections), "preview": str(preview)}
                rows.append(row)
                destination.write(json.dumps({**row, "detections": detections}) + "\n")
                print(f"{index}/{len(paths)} {path.name}: {len(detections)} cluster detections", flush=True)
    with (args.output / "counts.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {"checkpoint": str(args.checkpoint.resolve()), "class_name": "tomato_cluster",
               "threshold": args.threshold, "images": len(rows), "device": device,
               "run_utc": datetime.now(timezone.utc).isoformat(),
               "note": "Model detections of visual clusters; same-stem truss membership is not verified."}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Saved counts in {args.output.resolve()}")


if __name__ == "__main__":
    main()
