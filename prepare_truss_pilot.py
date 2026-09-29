"""Copy a fixed Laboro image sample into train, valid, and test folders."""

import argparse
import json
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

from inspect_dataset import load_split


def sequence_position(pair) -> tuple[str, int] | None:
    name = Path(pair[0]["file_name"]).stem
    dated = re.fullmatch(r"IMG_(\d{8})_(\d{6})", name)
    if dated:
        moment = datetime.strptime(dated.group(1) + dated.group(2), "%Y%m%d%H%M%S")
        return dated.group(1), moment.hour * 3600 + moment.minute * 60 + moment.second
    numbered = re.fullmatch(r"IMG_(\d+)", name)
    if numbered:
        return "numbered", int(numbered.group(1))
    return None


def too_close(first, second) -> bool:
    a = sequence_position(first)
    b = sequence_position(second)
    if a is None or b is None or a[0] != b[0]:
        return False
    minimum_gap = 30 if a[0] != "numbered" else 6
    return abs(a[1] - b[1]) < minimum_gap


def choose(items, count: int, rng: random.Random, blocked=(), space_within=True):
    shuffled = list(items)
    rng.shuffle(shuffled)
    selected = []
    for pair in shuffled:
        if any(too_close(pair, other) for other in blocked):
            continue
        if space_within and any(too_close(pair, other) for other in selected):
            continue
        selected.append(pair)
        if len(selected) == count:
            return sorted(selected, key=lambda item: item[0]["file_name"])
    raise ValueError(f"Could select only {len(selected)} of {count} spaced images; lower the sample size")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/laboro_tomato"))
    parser.add_argument("--output", type=Path, default=Path("data/truss_pilot"))
    parser.add_argument("--train", type=int, default=50)
    parser.add_argument("--valid", type=int, default=10)
    parser.add_argument("--test", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest-copy", type=Path, default=Path("truss_pilot_manifest.json"))
    args = parser.parse_args()
    if min(args.train, args.valid, args.test) < 1:
        parser.error("Each split needs at least one image")
    if args.output.exists():
        parser.error(f"{args.output} already exists; refusing to change a fixed split")

    _, train_pool = load_split(args.source, "train")
    _, test_pool = load_split(args.source, "test")
    train_pool = sorted(train_pool, key=lambda pair: pair[0]["file_name"])
    test_pool = sorted(test_pool, key=lambda pair: pair[0]["file_name"])
    seen_pairs = test_pool[:10]
    seen_test = {item[0]["file_name"] for item in seen_pairs}
    untouched_test_pool = [pair for pair in test_pool if not any(too_close(pair, seen) for seen in seen_pairs)]

    rng = random.Random(args.seed)
    selected_test = choose(untouched_test_pool, args.test, rng)
    selected_valid = choose(train_pool, args.valid, rng, blocked=selected_test)
    selected_train = choose(train_pool, args.train, rng, blocked=[*selected_test, *selected_valid], space_within=False)
    selected = {
        "train": selected_train,
        "valid": selected_valid,
        "test": selected_test,
    }
    manifest = {"source": str(args.source), "seed": args.seed, "excluded_seen_test": sorted(seen_test), "splits": {}}
    for split, pairs in selected.items():
        names = [Path(item["file_name"]).name for item, _ in pairs]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate base filenames in {split}; manual disambiguation required")
        manifest["splits"][split] = [
            {"source_image_id": item["id"], "source_file_name": item["file_name"], "file_name": Path(item["file_name"]).name}
            for item, _ in pairs
        ]

    manifest_text = json.dumps(manifest, indent=2) + "\n"
    if args.manifest_copy.exists() and args.manifest_copy.read_text(encoding="utf-8") != manifest_text:
        raise ValueError(f"{args.manifest_copy} disagrees with the new split; preserve or intentionally update it first")
    for split, pairs in selected.items():
        split_dir = args.output / split
        split_dir.mkdir(parents=True)
        for item, source_path in pairs:
            shutil.copy2(source_path, split_dir / Path(item["file_name"]).name)
    (args.output / "manifest.json").write_text(manifest_text, encoding="utf-8")
    args.manifest_copy.write_text(manifest_text, encoding="utf-8")
    print(f"Prepared {args.output}: {args.train} train, {args.valid} valid, {args.test} test images")
    print("The first ten previously viewed Laboro test images were excluded from the pilot test set.")
    print("Review the selected images before uploading them for annotation.")


if __name__ == "__main__":
    main()
