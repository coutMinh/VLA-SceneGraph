"""Success-rate overview figure for the README, generated only from committed result files.

python scripts/plot_results.py            # writes media/success_overview.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SPATIAL = ROOT / "results/spatial"
CROSS_SUITE = ROOT / "results/cross_suite/summary.json"
ROLE_COLOR = {"baseline": "#2a78d6", "control": "#1baf7a", "intervention": "#eb6834"}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def role(name):
    if name.endswith("E0"):
        return "baseline"
    return "control" if "ctrl" in name or name.endswith("lang") or name == "S1" else "intervention"


def rows():
    e0 = json.loads((SPATIAL / "e0_summary.json").read_text())
    part3 = json.loads((SPATIAL / "report.json").read_text())["conditions"]
    spatial = [("E0", e0["successes"], e0["completed"])]
    for name in ("S1", "S2-ctrl", "S2", "T1-ctrl", "T1"):
        s = part3[name]["summary"]
        spatial.append((name, s["successes"], s["completed"]))
    groups = [("LIBERO-Spatial", spatial)]
    titles = {
        "libero_object": "LIBERO-Object",
        "libero_goal": "LIBERO-Goal",
        "libero_10": "LIBERO-Long",
    }
    xs = json.loads(CROSS_SUITE.read_text())["conditions"]
    for suite, title in titles.items():
        groups.append(
            (title, [(r["condition"], r["successes"], r["n"]) for r in xs if r["suite"] == suite])
        )
    return groups


def main():
    groups = rows()
    fig, axes = plt.subplots(
        1,
        len(groups),
        figsize=(14, 3.6),
        sharey=True,
        facecolor=SURFACE,
        gridspec_kw={"width_ratios": [len(g[1]) for g in groups]},
    )
    for ax, (title, data) in zip(axes, groups):
        ax.set_facecolor(SURFACE)
        names = [d[0] for d in data]
        rates = [d[1] / d[2] for d in data]
        ax.bar(
            range(len(data)),
            rates,
            width=0.7,
            color=[ROLE_COLOR[role(n)] for n in names],
            edgecolor=SURFACE,
            linewidth=2,
        )
        for i, (_, k, n) in enumerate(data):
            ax.text(i, k / n + 0.02, f"{k}/{n}", ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_xticks(range(len(data)))
        ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8, color=MUTED)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_ylim(0, 1.12)
        ax.yaxis.grid(True, color="#e6e5e1", linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.tick_params(axis="y", colors=MUTED, labelsize=8, length=0)
    axes[0].set_ylabel("Task success rate", color=MUTED, fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ROLE_COLOR.values()]
    labels = [
        "official baseline",
        "control / language-only change",
        "spatial or temporal intervention",
    ]
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.02),
        labelcolor=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = ROOT / "media/success_overview.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print(out)


if __name__ == "__main__":
    main()
