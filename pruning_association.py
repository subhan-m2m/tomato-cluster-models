"""Versioned geometric associations, never verified biological truss membership.

No model or imaging dependency: the service and offline review use this same code.
Coordinates are xyxy pixels in the oriented evidence image.
"""
from copy import deepcopy
import hashlib
import json
import math

VERSION = "visible-association-v1"
SETTINGS = {"margin": 0.0, "min_coverage": 0.5, "separation": 0.15,
            "fruit_duplicate_iou": 0.95, "cluster_overlap_iou": 0.55}


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection(a, b):
    return area([max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])])


def iou(a, b):
    shared = intersection(a, b)
    return shared / max(1e-9, area(a) + area(b) - shared)


def stable_detections(detections, prefix, width, height):
    """Reject degenerate boxes, clip geometry, and give a deterministic display order."""
    result = []
    for detection in detections:
        box = detection["box"]
        if len(box) != 4 or not all(math.isfinite(v) for v in [*box, detection["score"]]):
            raise ValueError("Non-finite detection")
        clipped = [max(0, min(width, box[0])), max(0, min(height, box[1])),
                   max(0, min(width, box[2])), max(0, min(height, box[3]))]
        if area(clipped) <= 0:
            continue
        key = json.dumps([prefix, box, detection["score"]], separators=(",", ":"))
        result.append({**detection, "box": clipped,
                       "id": prefix + "-" + hashlib.sha256(key.encode()).hexdigest()[:12]})
    # Coalesce byte-identical predictions before IDs are used as keys.
    unique = {d["id"]: d for d in result}
    return sorted(unique.values(), key=lambda d: (d["box"][1], d["box"][0], d["id"]))


def associate(tomatoes, clusters, width, height, settings=None):
    cfg = {**SETTINGS, **(settings or {})}
    if not 0 <= cfg["margin"] <= 0.1:
        raise ValueError("Expansion must be bounded to 0–10% per side")
    fruit = stable_detections(tomatoes, "fruit", width, height)
    groups = stable_detections(clusters, "cluster", width, height)
    suppressed = []
    kept = []
    for f in sorted(fruit, key=lambda d: (-d["score"], d["id"])):
        duplicate = next((k for k in kept if iou(k["box"], f["box"]) >= cfg["fruit_duplicate_iou"]), None)
        if duplicate:
            suppressed.append({"fruit_id": f["id"], "kept_id": duplicate["id"], "reason": "near_identical_box"})
        else:
            kept.append(f)
    kept.sort(key=lambda d: (d["box"][1], d["box"][0], d["id"]))
    flags = {g["id"]: [] for g in groups}
    regions = {}
    for g in groups:
        x1, y1, x2, y2 = g["box"]
        dx, dy = (x2-x1)*cfg["margin"], (y2-y1)*cfg["margin"]
        regions[g["id"]] = [max(0, x1-dx), max(0, y1-dy), min(width, x2+dx), min(height, y2+dy)]
        if x1 <= 1 or y1 <= 1 or x2 >= width-1 or y2 >= height-1:
            flags[g["id"]].append("edge_clipped")
    for i, a in enumerate(groups):
        for b in groups[i+1:]:
            shared = intersection(a["box"], b["box"])
            if iou(a["box"], b["box"]) >= cfg["cluster_overlap_iou"] or shared/max(1e-9, min(area(a["box"]), area(b["box"]))) >= 0.9:
                flags[a["id"]].append("overlapping_or_merged_group")
                flags[b["id"]].append("overlapping_or_merged_group")
    memberships = []
    for f in kept:
        box = f["box"]
        cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        candidates = []
        for g in groups:
            region = regions[g["id"]]
            coverage = intersection(box, region)/area(box)
            if region[0] <= cx <= region[2] and region[1] <= cy <= region[3] and coverage >= cfg["min_coverage"]:
                original = g["box"]
                original_coverage = intersection(box, original)/area(box)
                supported = original[0] <= cx <= original[2] and original[1] <= cy <= original[3] and original_coverage >= cfg["min_coverage"]
                # Original containment outranks expansion-only support by a full point.
                score = (1 if supported else 0) + (original_coverage if supported else coverage)
                candidates.append({"cluster_id": g["id"], "score": round(score, 6),
                                   "coverage": round(coverage, 6), "reason": "original_box" if supported else "expanded_box"})
        candidates.sort(key=lambda c: (-c["score"], c["cluster_id"]))
        clear = candidates and (len(candidates) == 1 or candidates[0]["score"]-candidates[1]["score"] >= cfg["separation"])
        memberships.append({"fruit_id": f["id"], "cluster_id": candidates[0]["cluster_id"] if clear else None,
                            "status": "assigned" if clear else "ambiguous" if candidates else "unassigned",
                            "candidates": candidates,
                            "reason": "clear_geometric_match" if clear else "competing_groups" if candidates else "outside_candidate_regions"})
    return {"version": VERSION, "settings": cfg, "tomatoes": kept, "clusters": groups,
            "memberships": memberships, "quality_flags": {k: sorted(set(v)) for k, v in flags.items()},
            "suppressed_duplicates": suppressed, "candidate_regions": regions}


def assess(association, profile, complete=True):
    limit = profile["maximum"]
    results = []
    for group in association["clusters"]:
        gid = group["id"]
        assigned = [m["fruit_id"] for m in association["memberships"] if m["cluster_id"] == gid]
        ambiguous = [m["fruit_id"] for m in association["memberships"] if m["status"] == "ambiguous" and any(c["cluster_id"] == gid for c in m["candidates"])]
        flags = association["quality_flags"][gid]
        if not complete or limit is None:
            state = "not_evaluated"
        elif len(assigned) > limit:
            state = "over_limit"
        elif flags or len(assigned)+len(ambiguous) > limit:
            state = "uncertain"
        else:
            state = "no_flag"
        results.append({"cluster_id": gid, "assigned_fruit_ids": assigned, "ambiguous_fruit_ids": ambiguous,
                        "visible_count": len(assigned), "candidate_count": len(assigned)+len(ambiguous),
                        "maximum": limit, "excess": max(0, len(assigned)-limit) if limit is not None else None,
                        "state": state, "quality_flags": flags, "rule_version": profile["version"], "ripeness": None})
    return deepcopy(results)
