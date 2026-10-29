"""Identity-based placement evidence and explicitly provisional pose-based labels."""

from __future__ import annotations

import math
from pathlib import Path

from sgvla.eval.scenes import BOWL1, BOWL2
from sgvla.io import digest, write_json


def measure_trace(path, target):
    import numpy as np

    with np.load(path, allow_pickle=False) as trace:
        names = trace["object_names"].tolist()
        start = int(np.flatnonzero(trace["phase"] == 1)[0])
        placed = trace["bowl_on_plate"][start:].astype(bool)
        first = []
        evidence = {}
        for index, name in enumerate((BOWL1, BOWL2)):
            positions = trace["object_poses"][start:, names.index(name), :3]
            lifted = float((positions[:, 2] - positions[0, 2]).max())
            hits = np.flatnonzero(placed[:, index])
            first.append(int(hits[0] + start) if len(hits) else None)
            evidence[name] = {
                "max_lift_m": lifted,
                "max_xy_displacement_m": float(
                    np.linalg.norm(positions[:, :2] - positions[0, :2], axis=1).max()
                ),
                "first_on_plate_frame": first[-1],
                "ever_on_plate": bool(len(hits)),
            }
        if first[0] is None and first[1] is None:
            identity = None
        elif first[0] is not None and first[0] == first[1]:
            identity = "tie"
        elif first[1] is None or (first[0] is not None and first[0] < first[1]):
            identity = BOWL1
        else:
            identity = BOWL2
        other = BOWL2 if target == BOWL1 else BOWL1
        if evidence[target]["ever_on_plate"]:
            candidate = "target_placed"
        elif identity == other:
            candidate = "wrong_object_placed"
        elif evidence[other]["max_lift_m"] > 0.03 and evidence[target]["max_lift_m"] <= 0.03:
            candidate = "wrong_object_lift_candidate"
        elif evidence[target]["max_lift_m"] > 0.03:
            candidate = "placement_failure_candidate"
        else:
            candidate = "grasp_or_reach_or_timeout_unresolved"
        return {
            "target_object": target,
            "first_placed": identity,
            "exclusive_first_target_success": identity == target,
            "wrong_first_placement": identity == other,
            "distractor_ever_placed": evidence[other]["ever_on_plate"],
            "pose_evidence": evidence,
            "provisional_label": candidate,
            "label_status": "requires_visual_validation",
            "policy_start_frame": start,
        }


def contact_sheet(video, output, stride=5):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw

    with imageio.get_reader(str(video)) as reader:
        frames = [frame for frame in reader]
    indices = sorted(set(list(range(0, len(frames), stride)) + [len(frames) - 1]))
    cols, size, label = 8, 144, 20
    sheet = Image.new(
        "RGB", (cols * size, math.ceil(len(indices) / cols) * (size + label)), "white"
    )
    draw = ImageDraw.Draw(sheet)
    for k, frame_id in enumerate(indices):
        x, y = (k % cols) * size, (k // cols) * (size + label)
        image = Image.fromarray(frames[frame_id]).resize((size, size))
        sheet.paste(image, (x, y + label))
        draw.text((x + 2, y + 2), str(frame_id), fill="black")
    sheet.save(output, quality=80)
    return indices


def audit_e0(runtime, output):
    from sgvla.analysis.baseline import load_baseline

    manifest, root, summary, done = load_baseline(runtime)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for entry in manifest["episodes"]:
        _, metrics, folder = done[entry["episode_id"]]
        evidence = measure_trace(folder / "trajectory.npz", entry["target_object"])
        rows.append(
            {
                "episode_id": entry["episode_id"],
                "success": metrics["success"],
                "steps": metrics["steps"],
                "evidence": evidence,
                "completion_sha256": digest(
                    root / "episodes" / entry["episode_id"] / "complete.json"
                ),
            }
        )
    failed = [row["episode_id"] for row in rows if not row["success"]]
    matched = ["E0-t" + key.split("-t")[1].split("-s")[0] + "-s00" for key in failed]
    selection = list(dict.fromkeys(failed + matched))
    sheets = {}
    for episode_id in selection:
        folder = done[episode_id][2]
        sheets[episode_id] = {}
        for camera in ("agentview", "wrist"):
            image = output / (episode_id + "-" + camera + ".jpg")
            frames = contact_sheet(folder / (camera + ".mp4"), image)
            sheets[episode_id][camera] = {
                "path": str(image),
                "frames": frames,
                "sha256": digest(image),
            }
    report = {
        "baseline_context": summary["context"],
        "rows": rows,
        "visual_review_selection": selection,
        "contact_sheets": sheets,
        "manual_review": "pending; contact sheets are generated evidence, not a completed review",
    }
    write_json(output / "audit.json", report)
    return report
