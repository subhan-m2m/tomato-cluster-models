"""Build a standalone, offline team dashboard from recorded metrics and previews."""

import base64
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DESTINATION = ROOT / "presentation"


def read_json(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def main():
    original = read_json("results/agrob/test_tomato_summary.json")
    fruit_valid = read_json("results/agrob_finetune_v1_480/validation_summary.json")
    fruit_development = read_json("results/agrob_finetune_v1_480/development_summary.json")
    cluster_valid = read_json("results/agrob_clusters_v1/validation_summary.json")
    cluster_test = read_json("results/agrob_clusters_v1/test_summary.json")
    source_paths = [
        "results/agrob/test_tomato_summary.json",
        "results/agrob_finetune_v1_480/validation_summary.json",
        "results/agrob_finetune_v1_480/development_summary.json",
        "results/agrob_finetune_v1_480/run_config.json",
        "results/agrob_clusters_v1/validation_summary.json",
        "results/agrob_clusters_v1/test_summary.json",
        "results/agrob_clusters_v1/run_config.json",
        "results/agrob_clusters_v1/dataset_summary.json",
        "results/agrob_clusters_v1/selected_settings.json",
    ]
    def count_rows(path):
        source_paths.append(path)
        with (ROOT / path).open(encoding="utf-8") as stream:
            return {row["file_name"]: row for row in csv.DictReader(stream)}
    fruit_rows = count_rows("results/agrob_finetune_v1_480/development_counts.csv")
    cluster_rows = count_rows("results/agrob_clusters_v1/test_counts.csv")
    samples = []
    def add_sample(sample_id, title, filename, preview_path, unit, rows, issue, note, threshold):
        row = rows[filename]
        content = (ROOT / preview_path).read_bytes()
        source_paths.append(preview_path)
        samples.append({"id": sample_id, "title": title, "filename": filename, "unit": unit,
                        "labeled": int(row[f"ground_truth_{unit}_count"]),
                        "predicted": int(row[f"predicted_{unit}_count"]),
                        "matched": int(row[f"matched_{unit}_iou_50"]),
                        "issue": issue, "note": note, "threshold": threshold,
                        "image": "data:image/jpeg;base64," + base64.b64encode(content).decode("ascii")})
    fruit_name = "tomates_2020-08-06-11-35-15_side_0068.jpg"
    before_row = {fruit_name: {"ground_truth_fruit_count": fruit_rows[fruit_name]["ground_truth_fruit_count"],
                              "predicted_fruit_count": 0, "matched_fruit_iou_50": 0}}
    assert original["predicted_fruit"] == 0
    add_sample("original", "Original detector · side 0068", fruit_name,
               "outputs/agrob-fruit-detr-test-tomato-t030/previews/" + fruit_name,
               "fruit", before_row, "No Tomato-class detections",
               "The green boxes show five source-labeled fruit. The original Tomato class produced no detections on this frame or across the 152-image sequence.", 0.3)
    add_sample("fruit-clear", "Fine-tuned detector · side 0068", fruit_name,
               "outputs/agrob-gpu-full-unpadded-v1-480-development/previews/" + fruit_name,
               "fruit", fruit_rows, "Clear fruit recovered",
               "The fine-tuned model matches all five labeled fruit and adds one extra box. This is a selected example of improvement, not the average result.", 0.9)
    fruit_name = "tomates_2020-08-06-11-35-15_side_0080.jpg"
    add_sample("fruit-dense", "Fine-tuned detector · side 0080", fruit_name,
               "outputs/agrob-gpu-full-unpadded-v1-480-development/previews/" + fruit_name,
               "fruit", fruit_rows, "Dense groups and possible missing labels",
               "Repeated boxes appear around dense fruit at the top left. Other red boxes appear on visible fruit without source labels. A human review must distinguish duplicate predictions from missing annotations.", 0.9)
    for sample_id, title, filename, preview, issue, note in (
        ("cluster-hidden", "Cluster detector · Barroselas 0083",
         "tomate_barroselas_20200806_0083_jpg.rf.BadabIsj60q7DhRt6fkr.jpg", "barroselas_0083.jpg",
         "Shaded and partly hidden groups missed",
         "Larger clear groups are recovered; several small, shaded, occluded, and edge groups are missed. The model undercounts this image by nine clusters."),
        ("cluster-split", "Cluster detector · side 0137",
         "tomates_2020-08-06-11-35-15_side_0137_jpg.rf.7oDn4TeUQ9TtGjY5Bbqa.jpg", "side_0137.jpg",
         "One group becomes several predictions",
         "Several red boxes split a large green-labeled cluster into smaller detections. Other predicted groups lie outside the supplied boxes and need a label-completeness check."),
        ("cluster-partial", "Cluster detector · Barroselas 0177",
         "tomate_barroselas_20200806_0177_jpg.rf.8YXLxP68VuATn0xuqQ7u.jpg", "barroselas_0177.jpg",
         "Partial group boundaries",
         "Large and occluded groups are missed. The two predictions cover only parts of labeled clusters and do not meet the box overlap criterion."),
    ):
        add_sample(sample_id, title, filename, "results/agrob_clusters_v1/reviewed_previews/" + preview,
                   "cluster", cluster_rows, issue, note, 0.9)
    assert set(fruit_rows) == {Path(path).name for path in (ROOT / "outputs/agrob-fruit-detr-test-tomato-t030/previews").glob("*.jpg")}
    assert sum(int(row["ground_truth_fruit_count"]) for row in fruit_rows.values()) == original["ground_truth_fruit"] == 1842
    assert len(cluster_rows) == cluster_test["chosen_metrics"]["images"] == 44
    data = {"original": original, "fruitValid": fruit_valid["chosen_metrics"],
            "fruitDevelopment": fruit_development["chosen_metrics"],
            "clusterValid": cluster_valid["chosen_metrics"], "clusterTest": cluster_test["chosen_metrics"],
            "fruitRun": read_json("results/agrob_finetune_v1_480/run_config.json"),
            "clusterRun": read_json("results/agrob_clusters_v1/run_config.json"),
            "clusterData": read_json("results/agrob_clusters_v1/dataset_summary.json"),
            "samples": samples}
    template = (DESTINATION / "dashboard_template.html").read_text(encoding="utf-8")
    assert template.count("@@DATA@@") == 1
    content = template.replace("@@DATA@@", json.dumps(data, separators=(",", ":")).replace("<", "\\u003c"))
    output = DESTINATION / "tomato_findings_dashboard.html"
    output.write_text(content, encoding="utf-8", newline="\n")
    provenance = {"description": "Saved metrics and selected source previews embedded in the standalone dashboard",
                  "generated_file": output.name, "sample_images": len(samples), "sources": []}
    for name in sorted(set(source_paths)):
        provenance["sources"].append({"path": name, "sha256": hashlib.sha256((ROOT / name).read_bytes()).hexdigest()})
    (DESTINATION / "dashboard_sources.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"Built {output} ({output.stat().st_size / 1024**2:.2f} MiB); embedded {len(samples)} images")


if __name__ == "__main__":
    main()
