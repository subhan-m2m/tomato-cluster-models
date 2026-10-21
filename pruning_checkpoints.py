"""Verify or package the two trained checkpoints required by the local prototype.

Standard-library only. Checkpoints are distributed separately from Git. Archive
members are relative to this model repository and contain no photos or notes.
"""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_STORED

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "pruning-checkpoints.json"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    problems = []
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            problems.append(f'Missing: {entry["path"]}')
        elif path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            problems.append(f'Checksum mismatch: {entry["path"]}')
    if problems:
        raise ValueError("\n".join(problems) + "\nObtain the pruning checkpoint bundle from the maintainer and extract it into tomato-cluster-models. See LOCAL_SETUP.md.")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("verify", "package"))
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/pruning-checkpoints-v1.zip")
    args = parser.parse_args()
    try:
        if args.action == "package" and (args.output.exists() or args.output.with_suffix(".zip.sha256").exists()):
            raise ValueError("Choose a new output filename; existing bundles are never overwritten.")
        manifest = verify()
        if args.action == "verify":
            print(f'Verified {len(manifest["files"])} checkpoint files against pruning-checkpoints.json.')
            return
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(args.output, "w", compression=ZIP_STORED) as archive:
            for entry in manifest["files"]:
                archive.write(ROOT / entry["path"], entry["path"])
        checksum = digest(args.output)
        args.output.with_suffix(".zip.sha256").write_text(checksum + "  " + args.output.name + "\n", encoding="utf-8")
        print(f"Saved {args.output.resolve()} ({args.output.stat().st_size / 1024**2:.1f} MiB)")
        print("SHA-256: " + checksum)
    except (ValueError, OSError) as error:
        parser.exit(1, str(error) + "\n")


if __name__ == "__main__":
    main()
