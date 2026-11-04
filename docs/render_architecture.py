"""Render the repository architecture for portfolio and interview use."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def main():
    """Save a high-resolution PNG and editable SVG beside this script."""
    fig, ax = plt.subplots(figsize=(20, 12))
    fig.patch.set_facecolor("#f3f6fb")
    ax.set(xlim=(0, 20), ylim=(0, 12))
    ax.axis("off")
    ink, muted = "#14243b", "#52627a"
    ax.text(0.6, 11.3, "AdaptVLM", fontsize=32, weight="bold", color=ink)
    ax.text(
        0.6,
        10.83,
        "Human feedback → multimodal adaptation → measured evaluation",
        fontsize=17,
        color=muted,
    )
    ax.text(19.4, 11.35, "SYSTEM ARCHITECTURE", ha="right", fontsize=11, color=muted, weight="bold")

    def box(x, y, title, body, color, width=4.2, height=1.65):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.02,rounding_size=0.13",
                facecolor="white",
                edgecolor="#d6dfeb",
                linewidth=1.2,
            )
        )
        ax.plot([x + 0.2, x + 0.2], [y + 0.25, y + height - 0.25], color=color, linewidth=4)
        ax.text(x + 0.4, y + height - 0.39, title, fontsize=14, weight="bold", color=ink)
        ax.text(
            x + 0.4, y + height - 0.77, body, fontsize=10.8, color=muted, va="top", linespacing=1.55
        )

    def arrow(points, label=None, at=None, dashed=False):
        for i in range(len(points) - 1):
            ax.add_patch(
                FancyArrowPatch(
                    points[i],
                    points[i + 1],
                    arrowstyle="-|>" if i == len(points) - 2 else "-",
                    mutation_scale=14,
                    linewidth=1.5,
                    color="#6b7e99",
                    linestyle="--" if dashed else "-",
                    shrinkA=0,
                    shrinkB=0,
                )
            )
        if label:
            ax.text(
                *at,
                label,
                fontsize=10.5,
                color=muted,
                ha="center",
                bbox=dict(facecolor="#f3f6fb", edgecolor="none", pad=2),
            )

    xs = [0.6, 5.45, 10.3, 15.15]
    blue, teal, purple = "#3676d9", "#078a85", "#7a56c1"
    ax.text(0.6, 10.15, "01  INTERACT & CAPTURE", fontsize=11, weight="bold", color=blue)
    box(
        xs[0],
        8.15,
        "Gradio application",
        "Image + visual question\nCorrections + randomized A/B votes",
        blue,
    )
    box(
        xs[1],
        8.15,
        "Inference engine",
        "Reusable model + processor\nSmolVLM base / loaded LoRA adapter",
        blue,
    )
    box(
        xs[2],
        8.15,
        "Feedback service",
        "Snapshots of reviewed responses\nModel revision + generation settings",
        blue,
    )
    box(
        xs[3],
        8.15,
        "SQLite + image files",
        "Requests, corrections, preferences\nSplit registry + image fingerprints",
        blue,
    )
    for i in range(3):
        arrow([(xs[i] + 4.2, 8.98), (xs[i + 1], 8.98)])

    ax.text(0.6, 7.15, "02  VALIDATE & ADAPT OFFLINE", fontsize=11, weight="bold", color=teal)
    box(
        xs[0],
        5.15,
        "Data eligibility gate",
        "Train-only validated feedback\nExclude held-out / mock / synthetic",
        teal,
    )
    box(
        xs[1],
        5.15,
        "Two training datasets",
        "Human corrections → SFT examples\nDirectional votes → DPO pairs",
        teal,
    )
    box(
        xs[2],
        5.15,
        "LoRA training: SFT / DPO",
        "SFT: assistant-token supervision\nDPO: frozen-base reference (TRL)",
        teal,
    )
    box(
        xs[3],
        5.15,
        "Adapter artifacts",
        "Separate PEFT adapter weights\nConfig, data hashes + training metadata",
        teal,
    )
    for i in range(3):
        arrow([(xs[i] + 4.2, 5.98), (xs[i + 1], 5.98)])
    arrow(
        [(17.25, 8.15), (17.25, 7.55), (4.6, 7.55), (4.6, 6.8)],
        "Eligible export • validation repeated before training",
        (10, 7.65),
    )

    ax.text(0.6, 4.15, "03  EVALUATE & INSPECT", fontsize=11, weight="bold", color=purple)
    box(
        xs[0],
        2.15,
        "Dataset manifests",
        "VQAv2 local importer + test fixtures\nDisjoint train / validation / test splits",
        purple,
    )
    box(
        xs[1],
        2.15,
        "Matched evaluation",
        "Base vs adapter; guarded SFT vs DPO\nScores, categories + regressions",
        purple,
    )
    box(
        xs[2],
        2.15,
        "Efficiency benchmarks",
        "Separate processes; same conditions\nLatency, throughput, memory + size",
        purple,
    )
    box(
        xs[3],
        2.15,
        "Reports + dashboard",
        "JSON / CSV / plots + run provenance\nGradio experiment viewer",
        purple,
    )
    arrow([(4.8, 2.98), (5.45, 2.98)])
    arrow([(7.55, 2.15), (7.55, 1.75), (17.25, 1.75), (17.25, 2.15)])
    arrow([(14.5, 2.98), (15.15, 2.98)])
    arrow(
        [(17.25, 5.15), (17.25, 4.62), (7.55, 4.62), (7.55, 3.8)],
        "Load base + saved adapter for comparison",
        (12.2, 4.72),
    )
    arrow([(12.4, 4.62), (12.4, 3.8)])
    arrow([(4.6, 3.8), (4.6, 3.98), (12.4, 3.98)], dashed=True)
    ax.text(
        0.6,
        1.28,
        "Shared controls: YAML configs • fixed seeds • checkpoint / data hashes • explicit unsupported configurations",
        fontsize=12,
        color=ink,
    )
    ax.text(
        0.6,
        0.72,
        "VALIDATION: 65 tests passed; training and CPU benchmarks exercised on tiny random models + synthetic fixtures.",
        fontsize=11.5,
        weight="bold",
        color=teal,
    )
    ax.text(
        0.6,
        0.32,
        "Real-data adaptation gains and full-model GPU efficiency are not yet measured. DPO is experimental; adaptation is offline.",
        fontsize=11.5,
        color=muted,
    )
    out = Path(__file__).parent / "assets"
    out.mkdir(exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(
            out / f"adaptvlm-architecture.{extension}",
            dpi=180,
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.2,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
