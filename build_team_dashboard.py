"""Build an offline, photo-led team update from saved evaluation results."""

import base64
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "presentation"


def main():
    sources = []

    def read_json(name):
        sources.append(name)
        return json.loads((ROOT / name).read_text(encoding="utf-8"))

    def count_rows(name):
        sources.append(name)
        with (ROOT / name).open(encoding="utf-8") as stream:
            return {row["file_name"]: row for row in csv.DictReader(stream)}

    original = read_json("results/agrob/test_tomato_summary.json")
    fruit = read_json("results/agrob_finetune_v1_480/development_summary.json")["chosen_metrics"]
    v1 = read_json("results/agrob_clusters_v1/test_summary.json")["chosen_metrics"]
    v2 = read_json("results/agrob_clusters_v2/test_summary.json")["chosen_metrics"]
    comparison = read_json("results/agrob_clusters_v2/common_test_comparison.json")
    cluster_data = read_json("results/agrob_clusters_v2/dataset_summary.json")
    rows = {
        "original": count_rows("results/agrob/test_tomato_counts.csv"),
        "fruit": count_rows("results/agrob_finetune_v1_480/development_counts.csv"),
        "v1": count_rows("results/agrob_clusters_v1/test_counts.csv"),
        "v2": count_rows("results/agrob_clusters_v2/test_counts.csv"),
    }
    previews = {
        "original": "outputs/agrob-fruit-detr-test-tomato-t030/previews",
        "fruit": "outputs/agrob-gpu-full-unpadded-v1-480-development/previews",
        "v1": "outputs/agrob-cluster-detr-v1-480-test/previews",
        "v2": "outputs/agrob-cluster-detr-v2-480-test/previews",
    }
    manifests = {
        "v1": read_json("agrob_cluster_manifest.json"),
        "v2": read_json("agrob_cluster_v2_manifest.json"),
    }
    names = {version: {item["original_name"]: item["file_name"] for item in manifest["splits"]["test"]}
             for version, manifest in manifests.items()}
    samples = []

    def add_sample(version, case, original_name, title, note):
        filename = names[version][original_name] if version in names else original_name
        row = rows[version][filename]
        unit = "cluster" if version in names else "fruit"
        preview = f"{previews[version]}/{filename}"
        sources.append(preview)
        samples.append({
            "id": f"{version}-{case}", "case": case, "version": version,
            "title": title, "note": note, "filename": filename, "original_name": original_name,
            "unit": unit, "labeled": int(row[f"ground_truth_{unit}_count"]),
            "predicted": int(row[f"predicted_{unit}_count"]),
            "matched": int(row[f"matched_{unit}_iou_50"]),
            "image": "data:image/jpeg;base64," + base64.b64encode((ROOT / preview).read_bytes()).decode("ascii"),
        })

    # The same four fruit photos make the change easy to see.
    for case, title, note in (
        ("0068", "A smaller group of tomatoes", "Most marked tomatoes are found; one extra box remains."),
        ("0201", "Tomatoes among leaves", "The total is right here, although one miss and one extra box cancel out."),
        ("0084", "A wider greenhouse view", "The count is closer, but some tomatoes are still missed or counted extra."),
        ("0080", "A crowded view", "Too many boxes remain. Some fruit labels may also be missing and need review."),
    ):
        filename = f"tomates_2020-08-06-11-35-15_side_{case}.jpg"
        add_sample("original", case, filename, title, "Tomatoes are marked in green; this starting model found none.")
        add_sample("fruit", case, filename, title, note)

    # Identical images and labels for both cluster versions.
    group_cases = [
        ("0083", "A busy row", "tomate_barroselas_20200806_0083.jpg",
         "The newer model finds more of the marked groups. Some mistakes remain even when the total is close."),
        ("0137", "One group or several?", "tomates_2020-08-06-11-35-15_side_0137.jpg",
         "The newer count is closer. Group boundaries still need checking."),
        ("0177", "Partly hidden groups", "tomate_barroselas_20200806_0177.jpg",
         "More groups are found after the full annotations, but two are still missed."),
        ("0033", "A crowded row", "tomate_barroselas_20200806_0033.jpg",
         "The newer total matches the marked count; misses and extra boxes still cancel out."),
    ]
    for case, title, original_name, note in group_cases:
        for version in ("v1", "v2"):
            add_sample(version, case, original_name, title, note)
        first, second = samples[-2:]
        assert first["labeled"] == second["labeled"]

    for case, title, note in (
        ("0191", "A clear success", "All three marked groups are found in this selected example."),
        ("0044", "Groups still missed", "The model counts fewer groups than were marked."),
        ("0119", "Too many groups counted", "Extra group boxes make the count too high."),
        ("0150", "No groups in the photo", "The count correctly stays at zero here. More empty scenes need testing."),
    ):
        add_sample("v2", case, f"tomate_barroselas_20200806_{case}.jpg", title, note)

    assert set(rows["original"]) == set(rows["fruit"])
    assert original["predicted_fruit"] == 0
    assert len(rows["fruit"]) == 152 and len(rows["v1"]) == 44 and len(rows["v2"]) == 45
    assert len(samples) == 20
    data = {
        "original": original, "fruitDevelopment": fruit, "clusterV1": v1, "clusterTest": v2,
        "clusterComparison": comparison, "clusterData": cluster_data,
        "clusterTotals": {
            "images": sum(split["images"] for split in cluster_data["splits"].values()),
            "clusters": sum(split["clusters"] for split in cluster_data["splits"].values()),
        },
        "groupCases": [{"case": case, "title": title} for case, title, _, _ in group_cases],
        "sampleCount": len(samples), "samples": samples,
    }
    sources.extend(["presentation/dashboard_template.html", "presentation/PROJECT_CONTEXT.md"])
    template = (DESTINATION / "dashboard_template.html").read_text(encoding="utf-8")
    assert template.count("@@DATA@@") == 1
    output = DESTINATION / "tomato_findings_dashboard.html"
    output.write_text(template.replace("@@DATA@@", json.dumps(data, separators=(",", ":")).replace("<", "\\u003c")),
                      encoding="utf-8", newline="\n")
    provenance = {
        "description": "Saved counts and unmodified evaluation previews in a standalone team presentation",
        "generated_file": output.name, "sample_images": len(samples),
        "versions": {version: sum(sample["version"] == version for sample in samples) for version in rows},
        "sources": [{"path": name, "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()}
                    for name in sorted(set(sources))],
    }
    (DESTINATION / "dashboard_sources.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Built {output} ({output.stat().st_size / 1024**2:.2f} MiB); embedded {len(samples)} photos")


if __name__ == "__main__":
    main()
