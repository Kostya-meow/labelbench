"""Paired page-bootstrap analysis and standalone scientific figures."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from labelbench.robust_metrics import summarize
from labelbench.storage import write_json


def live_summary(pages: list[dict]) -> dict:
    methods, views, stability = defaultdict(list), defaultdict(list), defaultdict(list)
    for page in pages:
        if not page["complete"]:
            continue
        for key, row in page["stability"].items():
            stability[key].append(row)
        if not page["has_gt"]:
            continue
        for key, row in page["metrics"].items():
            methods[key].append(row)
        for key, row in page["view_metrics"].items():
            views[key].append(row)
    stable = {k: {field: float(np.mean([r[field] for r in rows])) for field in rows[0]}
              for k, rows in stability.items()}
    return {"methods": {k: summarize(rows) for k, rows in methods.items()},
            "views": {k: summarize(rows) for k, rows in views.items()}, "stability": stable,
            "evaluated_pages": sum(p["has_gt"] and p["complete"] for p in pages),
            "incomplete_pages": sum(not p["complete"] for p in pages),
            "unlabelled_pages": sum(not p["has_gt"] for p in pages)}


def _f1(counts: np.ndarray) -> np.ndarray:
    return 2*counts[..., 0] / np.maximum(2*counts[..., 0]+counts[..., 1]+counts[..., 2], 1)


def statistics(pages: list[dict], seed: int, samples: int) -> dict:
    valid = [p for p in pages if p["has_gt"] and p["complete"]]
    if len(valid) < 2:
        return {"pages": len(valid), "limitation": "At least two complete GT pages required", "methods": {}, "contrasts": {}}
    names = sorted(set.intersection(*(set(p["metrics"]) for p in valid)))
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(valid), size=(samples, len(valid)))
    counts = {name: np.asarray([[p["metrics"][name][k] for k in ("tp", "fp", "fn")] for p in valid]) for name in names}
    bootstrap = {name: _f1(values[indices].sum(axis=1)) for name, values in counts.items()}
    methods = {name: {"f1": float(_f1(counts[name].sum(axis=0))),
                      "ci95": np.quantile(values, [.025, .975]).tolist(),
                      "page_mean": float(np.mean(_f1(counts[name]))),
                      "page_sd": float(np.std(_f1(counts[name]), ddof=1))} for name, values in bootstrap.items()}
    contrasts = {}
    chosen = "joint_stable"
    if chosen in names:
        for baseline in [name for name in names if name.startswith("model:") or name in {"nms", "consensus_clean", "mask_vote"}]:
            effect = methods[chosen]["f1"]-methods[baseline]["f1"]
            swaps = rng.integers(2, size=(samples, len(valid), 1)).astype(bool)
            a, b = counts[chosen][None], counts[baseline][None]
            null = _f1(np.where(swaps, a, b).sum(axis=1)) - _f1(np.where(swaps, b, a).sum(axis=1))
            contrasts[baseline] = {"delta_f1": effect,
                                   "ci95": np.quantile(bootstrap[chosen]-bootstrap[baseline], [.025, .975]).tolist(),
                                   "p_permutation": float((1+np.sum(np.abs(null) >= abs(effect)))/(samples+1))}
        previous = 0.
        for rank, name in enumerate(sorted(contrasts, key=lambda k: contrasts[k]["p_permutation"])):
            previous = max(previous, min(1., (len(contrasts)-rank)*contrasts[name]["p_permutation"]))
            contrasts[name]["p_holm"] = previous
    return {"pages": len(valid), "samples": samples, "seed": seed, "methods": methods, "contrasts": contrasts,
            "limitation": "Paired page bootstrap, not independent-book inference. Correlated pages can make intervals optimistic; exploratory until document grouping is supplied."}


def export_analysis(directory: Path, pages: list[dict], seed: int, samples: int) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = live_summary(pages)
    stats = statistics(pages, seed, samples)
    report = {**summary, "statistics": stats}
    write_json(directory / "analysis.json", report)
    with (directory / "page-metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "method", "tp", "fp", "fn", "f1", "f1_75", "boundary_f1", "region_iou"])
        writer.writeheader()
        for page in pages:
            if page["complete"] and page["has_gt"]:
                for method, row in page["metrics"].items():
                    writer.writerow({"image": page["image"], "method": method, **{k: row[k] for k in writer.fieldnames[2:]}})
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    methods = summary["methods"]
    if methods:
        names = list(methods)
        fig, ax = plt.subplots(figsize=(11, max(5, len(names)*.32)))
        ax.barh(names, [methods[n]["f1"] for n in names], color="#2b7661")
        if stats["methods"]:
            for index, name in enumerate(names):
                low, high = stats["methods"][name]["ci95"]
                ax.plot([low, high], [index, index], color="#172e28", linewidth=1.5)
        ax.set(xlabel="Micro F1 at polygon IoU 0.50", xlim=(0, 1), title=f"Complete GT pages: {summary['evaluated_pages']}")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(figures / "method-f1.png", dpi=160)
        plt.close(fig)
        thresholds = [f"trust:{v:.1f}" for v in [.2, .4, .6, .8] if f"trust:{v:.1f}" in methods]
        if thresholds:
            total = methods["joint_medoid"]["tp"] + methods["joint_medoid"]["fp"]
            fig, ax = plt.subplots(figsize=(7, 4))
            coverage = [(methods[n]["tp"]+methods[n]["fp"])/max(total, 1) for n in thresholds]
            risk = [1-methods[n]["precision"] if methods[n]["tp"]+methods[n]["fp"] else np.nan for n in thresholds]
            ax.plot(coverage, risk, "o-", color="#2b7661")
            for x, y, name in zip(coverage, risk, thresholds):
                if np.isfinite(y):
                    ax.annotate(name, (x, y))
            ax.set(xlabel="Retained candidate fraction", ylabel="False discovery fraction (1 - precision)",
                   xlim=(0, 1), ylim=(0, 1), title="Selection risk versus coverage; not calibrated probabilities")
            fig.tight_layout()
            fig.savefig(figures / "trust-risk.png", dpi=160)
            plt.close(fig)
        keys = sorted(summary["views"])
        providers = sorted({k.split("|")[0] for k in keys})
        views = sorted({k.split("|")[1] for k in keys})
        values = np.asarray([[summary["views"].get(f"{p}|{v}", {}).get("f1", np.nan) for v in views] for p in providers])
        fig, ax = plt.subplots(figsize=(12, max(3, len(providers)*.7)))
        im = ax.imshow(values, vmin=0, vmax=1, aspect="auto", cmap="YlGnBu")
        ax.set_xticks(range(len(views)), views, rotation=35, ha="right")
        ax.set_yticks(range(len(providers)), providers)
        ax.set_title("Per-model corruption robustness: F1 against unchanged original GT")
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(figures / "corruption-f1.png", dpi=160)
        plt.close(fig)
    (directory / "analysis-report.md").write_text(
        f"# Robustness experiment\n\nEvaluated complete GT pages: {summary['evaluated_pages']}. "
        f"Unlabelled: {summary['unlabelled_pages']}. Technical incomplete: {summary['incomplete_pages']}.\n\n"
        "Primary outcome: instance micro F1 at polygon IoU 0.50; secondary: F1@0.75, boundary F1 with unmatched penalties, union-area IoU/Dice. "
        "Use paired differences and Holm-adjusted p-values in analysis.json; do not select the winner by p-value alone. "
        "Page resampling is exploratory: document/book dependence is unknown. No target training was performed. "
        "Stability and trust scores are heuristic evidence, not calibrated probabilities.\n\n"
        "See page-metrics.csv, analysis.json and figures/. Partial reports cover only completed pages.\n", encoding="utf-8")
    return report
