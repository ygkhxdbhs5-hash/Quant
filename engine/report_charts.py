"""Single-panel report charts designed to copy/download in one image.

Charts MUST be rendered from the same in-memory payload that produced the
text report (see run_topup_chase_report.py). Footer embeds repro fingerprint.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load_results(path_or_dict: Union[str, Path, dict]) -> dict:
    if isinstance(path_or_dict, dict):
        return path_or_dict
    return json.loads(Path(path_or_dict).read_text(encoding="utf-8"))


def render_topup_chase_dashboard(
    results: Union[str, Path, dict],
    out_path: Union[str, Path],
) -> Path:
    """One PNG with Task1 / Task2 / Task3 charts — copy or download once."""
    data = _load_results(results)
    t1 = data.get("task1_new_entry_vs_topup") or {}
    t2 = data.get("task2_cost_components") or {}
    comp: List[Dict[str, Any]] = list(data.get("comparison") or [])
    window = data.get("window") or {}
    fp = data.get("repro") or {}
    fp_id = fp.get("fingerprint_id") or data.get("fingerprint_id") or "unknown"
    title_win = f"{window.get('start', '?')} → {window.get('end', '?')}"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 10.5), facecolor="#f7f5f1")
    fig.suptitle(
        f"Baseline v1 — Top-up chase & cost diagnostics\n{title_win}  |  repro_id={fp_id}",
        fontsize=13,
        fontweight="bold",
        color="#1a1a1a",
        y=0.98,
    )

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    labels = ["new_entry", "top_up"]
    counts = [int(t1.get("n_new_entry") or 0), int(t1.get("n_top_up") or 0)]
    colors = ["#2f6f4e", "#b45309"]
    bars = ax1.bar(labels, counts, color=colors, width=0.55)
    n_buy = int(t1.get("n_buy_orders") or sum(counts))
    ax1.set_title(
        f"Task 1 — BUY order mix (n={n_buy:,})",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )
    ax1.set_ylabel("Order count")
    ax1.set_facecolor("#f7f5f1")
    for b, c in zip(bars, counts):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{c:,}\n({100 * c / max(sum(counts), 1):.1f}%)",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax1.set_ylim(0, max(counts) * 1.28 if counts else 1)

    cost_vals = [
        float(t1.get("buy_cost_new_entry") or 0),
        float(t1.get("buy_cost_top_up") or 0),
    ]
    bars2 = ax2.bar(labels, [v / 1e6 for v in cost_vals], color=colors, width=0.55)
    ax2.set_title("Task 1 — BUY cost $ (millions)", loc="left", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Cost $M")
    ax2.set_facecolor("#f7f5f1")
    for b, v in zip(bars2, cost_vals):
        ax2.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"${v/1e6:.2f}M",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax2.set_ylim(0, (max(cost_vals) / 1e6) * 1.28 if cost_vals else 1)

    all_f = t2.get("all_fills") or {}
    comps = [
        ("spread", float(all_f.get("spread_cost") or 0)),
        ("impact", float(all_f.get("impact_cost") or 0)),
        ("commission", float(all_f.get("commission_cost") or 0)),
        ("slippage", float(all_f.get("slippage_cost") or 0)),
    ]
    c_labels = [c[0] for c in comps]
    c_vals = [c[1] for c in comps]
    c_colors = ["#1d4e89", "#c2410c", "#57534e", "#78716c"]
    total = sum(c_vals) or 1.0
    ax3.pie(
        c_vals,
        labels=c_labels,
        colors=c_colors,
        autopct=lambda p: f"{p:.1f}%",
        startangle=90,
        textprops={"fontsize": 9},
    )
    ax3.set_title(
        f"Task 2 — Cost components (total ${total/1e6:.2f}M)",
        loc="left",
        fontsize=11,
        fontweight="bold",
    )

    if comp:
        short = []
        for r in comp:
            lab = str(r.get("label", "?"))
            lab = (
                lab.replace("Fixed CS v2 (chase ON)", "CS v2 chase")
                .replace("Fixed CS v2 + NO-CHASE", "CS v2 no-chase")
                .replace("Flat 10bps RT (5bps/side)", "Flat 10bps")
                .replace("Flat 30bps RT (15bps/side)", "Flat 30bps")
            )
            short.append(lab)
        cagrs = [100 * float(r.get("CAGR") or 0) for r in comp]
        costs_m = [float(r.get("total_cost_dollars") or 0) / 1e6 for r in comp]
        x = np.arange(len(short))
        w = 0.38
        ax4b = ax4.twinx()
        b_cagr = ax4.bar(x - w / 2, cagrs, w, color="#2f6f4e", label="CAGR %")
        b_cost = ax4b.bar(
            x + w / 2, costs_m, w, color="#9a3412", alpha=0.85, label="Cost $M"
        )
        ax4.axhline(0, color="#444", lw=0.8)
        ax4.set_xticks(x)
        ax4.set_xticklabels(short, rotation=12, ha="right", fontsize=8)
        ax4.set_ylabel("CAGR % (green)", color="#2f6f4e")
        ax4b.set_ylabel("Total cost $M (brown)", color="#9a3412")
        ax4.set_title(
            "Task 3 — Variants (green=CAGR, brown=Cost $M)",
            loc="left",
            fontsize=11,
            fontweight="bold",
        )
        ax4.set_facecolor("#f7f5f1")
        ax4.legend(
            [b_cagr, b_cost], ["CAGR %", "Cost $M"], loc="upper right", fontsize=8
        )
        for b, v in zip(b_cagr, cagrs):
            ax4.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + (0.35 if v >= 0 else -1.4),
                f"{v:+.2f}%",
                ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=7.5,
                color="#14532d",
                fontweight="bold",
            )
        for b, v in zip(b_cost, costs_m):
            ax4b.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.35,
                f"${v:.1f}M",
                ha="center",
                va="bottom",
                fontsize=7.5,
                color="#7c2d12",
                fontweight="bold",
            )
    else:
        ax4.text(0.5, 0.5, "No comparison rows", ha="center", va="center")
        ax4.set_axis_off()

    for ax in (ax1, ax2, ax4):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    git = str(fp.get("git_head") or "")[:10]
    dirty = "*" if fp.get("git_dirty") else ""
    fig.text(
        0.5,
        0.012,
        f"repro_id={fp_id}  git={git}{dirty}  |  "
        f"same payload as topup_chase_report.txt / topup_chase_results.json  |  "
        f"do not compare charts without matching repro_id",
        ha="center",
        fontsize=7.5,
        color="#44403c",
    )
    fig.tight_layout(rect=[0, 0.035, 1, 0.94])
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def render_from_json_file(
    json_path: Union[str, Path],
    out_path: Optional[Union[str, Path]] = None,
) -> Path:
    json_path = Path(json_path)
    data = _load_results(json_path)
    fp_id = (data.get("repro") or {}).get("fingerprint_id") or "unknown"
    if out_path is None:
        out_path = json_path.with_name(f"topup_chase_charts_{fp_id}.png")
    primary = render_topup_chase_dashboard(data, out_path)
    canonical = json_path.with_name("topup_chase_charts.png")
    if Path(out_path).resolve() != canonical.resolve():
        render_topup_chase_dashboard(data, canonical)
    return primary
