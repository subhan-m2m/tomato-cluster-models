"""Package a trained tomato DETR checkpoint and its metric provenance for sharing."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CODE_FILES = ("count_tomatoes.py", "finetune_agrob_fruit.py", "evaluate_agrob_fruit.py",
              "requirements.txt", "requirements-train.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Training output directory")
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True, help="Matching Markdown run report")
    parser.add_argument("--output", type=Path, required=True, help="A new ZIP file")
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    checkpoint = args.run / "best_model"
    processor = json.loads((checkpoint / "preprocessor_config.json").read_text(encoding="utf-8"))
    if processor.get("pad_size") is not None:
        parser.error("Checkpoint has fixed padding and should not be packaged")
    valid = json.loads((args.validation / "summary.json").read_text(encoding="utf-8"))
    development = json.loads((args.development / "summary.json").read_text(encoding="utf-8"))
    if valid["chosen_metrics"]["threshold"] != development["chosen_metrics"]["threshold"]:
        parser.error("Development threshold differs from validation")
    files = [(ROOT / name, name) for name in CODE_FILES]
    files += [(checkpoint / name, f"best_model/{name}")
              for name in ("config.json", "preprocessor_config.json", "model.safetensors")]
    files += [(args.run / name, f"run/{name}")
              for name in ("run_config.json", "training_history.csv", "best_epoch.json")]
    files += [(args.validation / name, f"validation/{name}")
              for name in ("summary.json", "thresholds.csv", "counts.csv", "error_analysis.json")]
    files += [(args.development / name, f"development/{name}")
              for name in ("summary.json", "counts.csv", "error_analysis.json", "worst_images.csv")]
    files.append((args.report, "RUN_REPORT.md"))
    for path, _ in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = {"description": "One-class AgRob tomato DETR checkpoint with evaluation provenance",
                "threshold": valid["chosen_metrics"]["threshold"], "files": []}
    usage = ("# Use this tomato detector\n\n"
             "Read RUN_REPORT.md for the measured AgRob results and limitations. "
             "After installing CUDA-enabled PyTorch and requirements-train.txt, run from this extracted folder:\n\n"
             "```powershell\n"
             f"python count_tomatoes.py --checkpoint best_model --threshold {manifest['threshold']} "
             "--images C:\\path\\to\\photo.jpg --output counts-for-photo\n"
             "```\n\n"
             "The output contains counts.csv and red-box previews. These are detections, "
             "so inspect the boxes before using the count. This package contains no AgRob images.\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w") as archive:
        for path, member in files:
            content = path.read_bytes()
            archive.writestr(member, content, compress_type=zipfile.ZIP_STORED)
            manifest["files"].append({"path": member, "bytes": len(content),
                                      "sha256": hashlib.sha256(content).hexdigest()})
        usage_bytes = usage.encode("utf-8")
        archive.writestr("USAGE.md", usage_bytes)
        manifest["files"].append({"path": "USAGE.md", "bytes": len(usage_bytes),
                                  "sha256": hashlib.sha256(usage_bytes).hexdigest()})
        archive.writestr("model_manifest.json", json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {args.output.resolve()} ({args.output.stat().st_size / 1024 ** 2:.1f} MiB)")


if __name__ == "__main__":
    main()
