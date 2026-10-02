"""Pack the prepared AgRob experiment for a GPU computer or Google Colab."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

from evaluate_agrob_fruit import DATA_ROOT


ROOT = Path(__file__).resolve().parent
DEFAULT_PREPARED = ROOT / "data/agrob_finetune_v1"
IMAGES = DATA_ROOT / "JPEGImages"
DEFAULT_OUTPUT = ROOT / "outputs/agrob-finetune-gpu-bundle.zip"
INCLUDED_FILES = (
    "AGROB_FINETUNE_GUIDE.md",
    "AGROB_FINETUNE_STATUS.md",
    "AGROB_GPU_COLAB.ipynb",
    "AGROB_LABEL_POLICY.md",
    "README.md",
    "agrob_finetune_split.json",
    "evaluate_agrob_fruit.py",
    "finetune_agrob_fruit.py",
    "convert_tomato_voc.py",
    "prepare_agrob_finetune.py",
    "requirements.txt",
    "requirements-train.txt",
)


def files_to_pack(prepared: Path) -> list[tuple[Path, str]]:
    files = [(ROOT / name, name) for name in INCLUDED_FILES]
    annotations = prepared / "annotations"
    names = set()
    for split in ("train", "valid", "development"):
        path = annotations / f"{split}.json"
        files.append((path, path.relative_to(ROOT).as_posix()))
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("categories") != [{"id": 0, "name": "tomato"}]:
            raise ValueError(f"Unexpected categories in {path}")
        for image in data["images"]:
            name = image["file_name"]
            if Path(name).name != name or name in names:
                raise ValueError(f"Unsafe or repeated image name: {name}")
            names.add(name)
    for name in sorted(names):
        image = IMAGES / name
        files.append((image, f"data/agrob/Dataset-Greenhouse_Tomato_AgRob/JPEGImages/{name}"))
    for path in (prepared / "summary.json", *(prepared / "review").glob("*")):
        files.append((path, path.relative_to(ROOT).as_posix()))
    for path, _ in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    prepared = args.prepared.resolve()
    if not prepared.is_relative_to(ROOT) or not prepared.is_relative_to(ROOT / "data"):
        parser.error("Prepared data must be inside this repository's data/ folder")
    files = files_to_pack(prepared)
    manifest = {"description": "AgRob one-class tomato DETR experiment; provisional source labels",
                "source": "AgRobTomato, Zenodo 5596799, CC BY 4.0",
                "files": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=2) as archive:
        for path, member in files:
            data = path.read_bytes()
            archive.writestr(member, data)
            manifest["files"].append({"path": member, "bytes": len(data),
                                      "sha256": hashlib.sha256(data).hexdigest()})
        archive.writestr("bundle_manifest.json", json.dumps(manifest, indent=2) + "\n")
    image_count = sum(member.startswith("data/agrob/") for _, member in files)
    print(f"Packed {len(files)} files ({image_count} images) in {args.output.resolve()}")
    print(f"Archive size: {args.output.stat().st_size / (1024 ** 2):.1f} MiB")


if __name__ == "__main__":
    main()
