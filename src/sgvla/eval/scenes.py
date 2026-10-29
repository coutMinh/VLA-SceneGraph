"""Read pinned LIBERO scenes/initial states without importing torch or MuJoCo."""

from __future__ import annotations

import ast
import io
import pickle
import re
import zipfile
from pathlib import Path

from sgvla.io import digest, write_json

BOWL1 = "akita_black_bowl_1"
BOWL2 = "akita_black_bowl_2"
PLATE = "plate_1"


def array_hash(value):
    import hashlib

    import numpy as np

    value = np.ascontiguousarray(value)
    header = f"{value.dtype.str}:{value.shape}:".encode()
    return hashlib.sha256(header + value.tobytes()).hexdigest()


class ArrayUnpickler(pickle.Unpickler):
    """Only the globals present in the pinned NumPy initial-state archives."""

    def find_class(self, module, name):
        import codecs

        import numpy as np

        allowed = {
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("_codecs", "encode"): codecs.encode,
        }
        if (module, name) not in allowed:
            raise ValueError(f"Unexpected initial-state pickle global: {module}.{name}")
        return allowed[module, name]


def read_initial_states(path):
    import numpy as np

    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(names) != 1:
            raise ValueError("Expected one data.pkl in a pinned LIBERO initial-state archive")
        states = ArrayUnpickler(io.BytesIO(archive.read(names[0]))).load()
    states = np.asarray(states)
    if states.ndim != 2 or len(states) < 5 or not np.isfinite(states).all():
        raise ValueError("Invalid LIBERO initial-state array")
    return states


def literal_assignment(path, name):
    for node in ast.parse(Path(path).read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing literal {name} in {path}")


def parse_bddl(text):
    tokens = re.findall(r"\(|\)|[^\s()]+", re.sub(r";[^\n]*", "", text))
    stack, roots = [], []
    for token in tokens:
        if token == "(":
            node = []
            (stack[-1] if stack else roots).append(node)
            stack.append(node)
        elif token == ")":
            if not stack:
                raise ValueError("Unbalanced BDDL")
            stack.pop()
        else:
            if not stack:
                raise ValueError("Unexpected BDDL token")
            stack[-1].append(token)
    if stack or len(roots) != 1:
        raise ValueError("Unbalanced BDDL")
    return {node[0]: node[1:] for node in roots[0] if isinstance(node, list)}


def typed_names(tokens):
    result, pending = {}, []
    for token in tokens:
        if pending and pending[-1] == "-":
            result.update({name: token for name in pending[:-1]})
            pending = []
        else:
            pending.append(token)
    if pending:
        raise ValueError("Untyped BDDL object")
    return result


def alternate_fragment(relation):
    region = relation[2]
    return {
        "main_table_next_to_ramekin_region": "next to the ramekin",
        "main_table_next_to_plate_region": "next to the plate",
        "main_table_next_to_box_region": "next to the cookie box",
        "wooden_cabinet_1_top_side": "on the wooden cabinet",
        "flat_stove_1_cook_region": "on the stove",
        "cookies_1": "on the cookie box",
    }[region]


def landmark_regions(sections, initial_relations):
    result = {}
    for name in (
        PLATE,
        "cookies_1",
        "glazed_rim_porcelain_ramekin_1",
        "wooden_cabinet_1",
        "flat_stove_1",
    ):
        relation = initial_relations[name]
        region = next(
            (r for r in sections[":regions"] if relation[2] == "main_table_" + r[0]), None
        )
        result[name] = {
            "initial_predicate": relation,
            "region_definition": region,
            "actual_world_pose": "pending_simulator",
        }
    return result


def verify_scenes(upstream):
    root = Path(upstream) / "third_party/libero/libero/libero"
    names = literal_assignment(root / "benchmark/libero_suite_task_map.py", "libero_task_map")[
        "libero_spatial"
    ]
    order = literal_assignment(root / "benchmark/__init__.py", "task_orders")[0]
    if len(names) != 10 or order != list(range(10)):
        raise ValueError("Pinned Spatial task order changed")
    rows = []
    for task_id, name in enumerate(names):
        bddl = root / "bddl_files/libero_spatial" / (name + ".bddl")
        initial = root / "init_files/libero_spatial" / (name + ".pruned_init")
        sections = parse_bddl(bddl.read_text(encoding="utf-8"))
        objects = typed_names(sections[":objects"])
        if objects.get(BOWL1) != "akita_black_bowl" or objects.get(BOWL2) != objects[BOWL1]:
            raise ValueError("Expected two bowls of the same asset class")
        goal = sections[":goal"]
        if goal != [["And", ["On", BOWL1, PLATE]]]:
            raise ValueError("Unexpected Spatial goal")
        initial_relations = {rel[1]: rel for rel in sections[":init"]}
        fragment = alternate_fragment(initial_relations[BOWL2])
        states = read_initial_states(initial)
        rows.append(
            {
                "task_id": task_id,
                "task_name": name,
                "official_instruction": name.replace("_", " "),
                "bddl_instruction": " ".join(sections[":language"]),
                "s1_instruction": f"pick up the black bowl {fragment} and place it on the plate",
                "e0_target": BOWL1,
                "s1_target": BOWL2,
                "object_types": objects,
                "fixture_types": typed_names(sections[":fixtures"]),
                "initial_relations": initial_relations,
                "regions": sections[":regions"],
                "landmarks": landmark_regions(sections, initial_relations),
                "s2_source_status": (
                    "candidate_requires_physics_and_uniqueness_checks"
                    if task_id in (0, 1, 6, 8)
                    else "table_center_not_landmark_controlled"
                    if task_id == 2
                    else "support_or_fixture_confound_requires_redesign"
                ),
                "swap_source_status": (
                    "both_bowls_on_table_candidate"
                    if task_id in (0, 1, 2, 8)
                    else "different_supports_or_drawer_unverified"
                ),
                "goal": goal,
                "bddl_sha256": digest(bddl),
                "init_file_sha256": digest(initial),
                "initial_state_count": len(states),
                "initial_state_width": states.shape[1],
                "state_sha256": [array_hash(state) for state in states],
                "runtime_checks": {
                    "s1_relation_uniqueness": "pending_simulator",
                    "s2_landmark_poses": "pending_simulator",
                    "swap_stability": "pending_simulator",
                },
            }
        )
    return {
        "schema": 1,
        "kind": "source_verification_only",
        "suite": "libero_spatial",
        "task_order_sha256": digest(root / "benchmark/libero_suite_task_map.py"),
        "official_client_sha256": digest(Path(upstream) / "examples/libero/main.py"),
        "tasks": rows,
    }


def write_report(report, path, markdown=None):
    write_json(path, report)
    if markdown is not None:
        lines = [
            "# Generated Spatial source verification",
            "",
            "Generated by `python -m sgvla eval-verify`. See SCENE_PROTOCOL.md for interpretation.",
            "No runtime physics, relation uniqueness or GPU evaluation is claimed.",
            "",
            "| Task | E0 instruction | S1 instruction (target bowl 2) | States | State width |",
            "|---|---|---|---|---|",
        ]
        for row in report["tasks"]:
            lines.append(
                f"| {row['task_id']} | {row['official_instruction']} | {row['s1_instruction']} "
                f"| {row['initial_state_count']} | {row['initial_state_width']} |"
            )
        lines += ["", f"Scene report SHA-256: `{digest(path)}`.", ""]
        lines += [
            "## Per-task targets and configured landmark placements",
            "",
            "Coordinates below are BDDL placement-region ranges, not measured world poses.",
            "E0 target is bowl 1; S1 target is bowl 2. Physics/uniqueness remain pending.",
            "",
            "| Task | Bowl 1 initial predicate | Bowl 2 initial predicate | S2 status "
            "| Swap source status |",
            "|---|---|---|---|---|",
        ]
        for row in report["tasks"]:
            lines.append(
                f"| {row['task_id']} | {' '.join(row['initial_relations'][BOWL1])} "
                f"| {' '.join(row['initial_relations'][BOWL2])} | {row['s2_source_status']} "
                f"| {row['swap_source_status']} |"
            )
        lines += [
            "",
            "| Task | Landmark | Region | BDDL ranges (xmin ymin xmax ymax) |",
            "|---|---|---|---|",
        ]
        for row in report["tasks"]:
            for name, info in row["landmarks"].items():
                region = info["region_definition"]
                ranges = (
                    next((x[1:] for x in region[1:] if x[0] == ":ranges"), []) if region else []
                )
                lines.append(
                    f"| {row['task_id']} | {name} | {info['initial_predicate'][2]} | {ranges} |"
                )
        lines.append("")
        Path(markdown).write_text("\n".join(lines), encoding="utf-8")
