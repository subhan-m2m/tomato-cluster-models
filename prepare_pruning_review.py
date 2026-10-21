"""Prepare a local manual membership/count review pack; no labels are invented.

Uses the sibling frontend's exact two-model coordinator to avoid a second detector
implementation. Predictions and photos stay in ignored outputs; settings and the
review protocol are tracked. Does not train models or modify existing datasets.
"""
import argparse
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import sys

from pruning_association import SETTINGS, VERSION, associate


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frontend", type=Path, default=root.parent/"frontend/agritwin-harvest-dashboard-oct-2026")
    parser.add_argument("--per-scene", type=int, default=12)
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.per_scene <= 20:
        parser.error("Choose a new output directory and 1–20 examples per scene.")
    sys.path.insert(0, str(args.frontend/"inference"))
    from server import Counter, decode_image
    import torch
    from PIL import ImageDraw

    manifest = json.loads((root/"agrob_cluster_v2_manifest.json").read_text())
    coco = json.loads((root/"data/agrob_clusters_v2/annotations/valid.json").read_text())
    annotations = {}
    for label in coco["annotations"]:
        annotations.setdefault(label["image_id"], []).append(label)
    images_by_name = {image["file_name"]: image for image in coco["images"]}
    candidates = manifest["splits"].get("valid", manifest["splits"].get("validation", []))
    counter = Counter(root/"outputs/agrob-gpu-full-unpadded-v1-480/best_model",
                      root/"outputs/agrob-cluster-detr-v2-480/best_model",
                      "cuda" if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True)
    samples = []
    for scene, prefix, partition in [("barroselas", "tomate_barroselas", "development"),
                                      ("side-camera", "tomates_", "reserved_evaluation")]:
        scene_images = sorted([i for i in candidates if i["file_name"].startswith(prefix)], key=lambda i: i["file_name"])
        n = min(args.per_scene, len(scene_images))
        selected = [scene_images[round(i*(len(scene_images)-1)/max(1, n-1))] for i in range(n)]
        for item in selected:
            source = root/"data/agrob_clusters_v2/images"/item["file_name"]
            image = decode_image(source.read_bytes())
            result = counter.count(image)
            if result["status"] != "complete":
                raise RuntimeError(f"Incomplete inference for {source.name}; inspect the local model error.")
            association = associate(result["tomatoes"]["detections"], result["clusters"]["detections"], image.width, image.height)
            image_id = images_by_name[item["file_name"]]["id"]
            reference_clusters = []
            for index, label in enumerate(annotations.get(image_id, []), 1):
                x, y, w, h = label["bbox"]
                reference_clusters.append({"id": f"reference-{index}", "box": [x, y, x+w, y+h], "visible_fruit_count": None,
                                           "quality": [], "source": "existing_cluster_annotation"})
            sample = {"file_name": item["file_name"], "sha256": item["sha256"], "scene": scene, "partition": partition,
                      "reviewed": False, "reviewer": "", "notes": "", "association": association,
                      "reference_clusters": reference_clusters,
                      "memberships": [{"fruit_id": f["id"], "reference_cluster_id": None, "status": "unreviewed"} for f in association["tomatoes"]],
                      "inference_seconds": result["inference_seconds"], "peak_gpu_bytes": result["peak_gpu_bytes"]}
            samples.append(sample)
            preview = image.copy()
            draw = ImageDraw.Draw(preview)
            for index, group in enumerate(reference_clusters, 1):
                draw.rectangle(group["box"], outline="#ffff00", width=2)
                draw.text((group["box"][0], group["box"][1]), f"R{index}", fill="#ffff00", stroke_width=1, stroke_fill="black")
            for index, group in enumerate(association["clusters"], 1):
                draw.rectangle(group["box"], outline="#2c9aff", width=3)
                draw.text((group["box"][0], group["box"][1]+15), f"C{index}", fill="white", stroke_width=1, stroke_fill="black")
            for index, fruit in enumerate(association["tomatoes"], 1):
                draw.rectangle(fruit["box"], outline="#ff4f4f", width=2)
                draw.text((fruit["box"][0], fruit["box"][1]), f"F{index}", fill="white", stroke_width=1, stroke_fill="black")
            preview.save(args.output/f"{len(samples):02d}.jpg", quality=92)
            image.save(args.output/f"{len(samples):02d}-source.jpg", quality=95)
            print(f'{len(samples)} {partition} {item["file_name"]}: {len(association["tomatoes"])} fruit / {len(association["clusters"])} groups', flush=True)
    reference = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "association_version": VERSION,
                 "settings": SETTINGS, "models": counter.versions, "review_status": "unreviewed", "samples": samples}
    (args.output/"reference.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    cards = [f'<section><h2>{i:02d} · {html.escape(s["partition"])} · {html.escape(s["scene"])}</h2><p>{html.escape(s["file_name"])}</p><img src="{i:02d}.jpg"><p>R = existing reference cluster (yellow), C = predicted cluster (blue), F = detected fruit (red). Membership is unreviewed.</p><details><summary>IDs and candidate memberships</summary><pre>{html.escape(json.dumps(s, indent=2))}</pre></details></section>' for i, s in enumerate(samples, 1)]
    (args.output/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Pruning membership review</title><style>body{font:15px system-ui;max-width:1400px;margin:30px auto;padding:20px;background:#f4f7f5;color:#172b32}section{background:white;padding:20px;margin:20px 0;border:1px solid #ccd6d0}img{max-width:100%}pre{white-space:pre-wrap;font-size:12px}</style><h1>Pruning membership review — labels required</h1><p>Edit reference.json after inspecting source photos. Do not use inferred memberships as ground truth. Keep reserved scene labels sealed until development choices are frozen. Existing video splits are not a fresh final benchmark.</p>'+''.join(cards), encoding="utf-8")
    print(f"Saved {len(samples)} unreviewed examples to {args.output.resolve()}")


if __name__ == "__main__":
    main()
