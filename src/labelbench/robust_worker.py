"""Standalone resumable experiment runner, independent of browser/server lifecycle."""

import argparse
import hashlib
import os
import random
import tempfile
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from labelbench.contracts import Annotation, safe_child_path
from labelbench.datasets import DatasetStore
from labelbench.geometry import overlap_matrix
from labelbench.robust_analysis import export_analysis, live_summary
from labelbench.robust_config import RobustConfig
from labelbench.robust_db import RobustDB
from labelbench.robust_fusion import combine
from labelbench.robust_inference import InferencePool
from labelbench.robust_metrics import measure
from labelbench.robust_transforms import restore, transform
from labelbench.settings import Settings


def run(db_path: Path, identifier: str) -> None:
    db = RobustDB(db_path)
    item = db.get(identifier)
    cfg = RobustConfig.model_validate(item["config"])
    settings = replace(Settings.from_environment(), device="cuda")
    directory = db_path.parent / identifier
    directory.mkdir(exist_ok=True)
    lock = (directory / "worker.lock").open("a+b")
    lock.seek(0)
    if os.name == "nt":
        import msvcrt

        if not lock.read(1):
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    import torch

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    started, previous_elapsed = time.monotonic(), item["elapsed"]
    stop_heartbeat = threading.Event()

    def heartbeat() -> None:
        while not stop_heartbeat.wait(3):
            db.update(identifier, heartbeat=time.time(), elapsed=previous_elapsed + time.monotonic()-started)

    db.update(identifier, status="running", pid=os.getpid(), heartbeat=time.time())
    ticker = threading.Thread(target=heartbeat, daemon=True)
    ticker.start()
    if os.name == "nt":
        import ctypes

        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    datasets = DatasetStore(settings.output_dir)
    pool = InferencePool(settings, cfg, directory)
    metric_pages = {p["image"]: p for p in db.metric_pages(identifier)}
    manifest = item["manifest"]
    consecutive_errors = 0
    try:
        for page_index, name in enumerate(manifest["images"]):
            if db.cancelled(identifier):
                raise InterruptedError("Cancelled at a durable checkpoint")
            if name in metric_pages and metric_pages[name]["complete"]:
                continue
            path = safe_child_path(Path(manifest["root"]), name)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != manifest["image_sha256"][name]:
                raise ValueError(f"Source image changed: {name}; create a new experiment")
            candidates, view_rows, timings, errors = [], {}, {}, []
            with Image.open(path) as source, tempfile.TemporaryDirectory(prefix="view-", dir=directory) as temporary:
                image = source.convert("RGB")
                size = image.size
                temporary_path = Path(temporary) / "current.png"
                for view in cfg.views:
                    pending = [p for p in cfg.providers if (db.task(identifier, name, p, view.key) or {}).get("status") != "done"]
                    if pending:
                        augmented, inverse = transform(image, view, cfg.max_side, cfg.seed, digest)
                        augmented.save(temporary_path, compress_level=1)
                    for provider in cfg.providers:
                        if db.cancelled(identifier):
                            raise InterruptedError("Cancelled; completed model/view checkpoints preserved")
                        key = f"{provider}|{view.key}"
                        cached = db.task(identifier, name, provider, view.key)
                        if cached and cached["status"] == "done":
                            rows = [Annotation.model_validate(a) for a in cached["result"]["annotations"]]
                            elapsed = cached["elapsed"]
                        else:
                            db.event(identifier, f"{page_index+1}/{len(manifest['images'])} · {name} · {provider} · {view.key}")
                            tick = time.monotonic()
                            try:
                                raw = pool.predict(provider, temporary_path)
                                rows = restore(raw, inverse, size, provider, view.key)
                                elapsed = time.monotonic()-tick
                                db.save_task(identifier, name, provider, view.key, elapsed,
                                             {"annotations": [a.model_dump(exclude={"mask_rle"}) for a in rows],
                                              "raw_count": len(raw), "discarded_geometry": len(raw)-len(rows)})
                            except Exception as error:  # noqa: BLE001 - persist failures without fake empty predictions
                                detail = f"{type(error).__name__}: {error}"
                                db.save_task(identifier, name, provider, view.key, time.monotonic()-tick, None, detail)
                                db.event(identifier, f"ERROR {name} {key}: {detail}")
                                errors.append(key)
                                continue
                        candidates.extend(rows)
                        view_rows[key], timings[key] = rows, elapsed
            has_gt = name in manifest["gt_images"]
            complete = not errors
            metrics, view_metrics, stability, outputs, evidence = {}, {}, {}, {}, []
            if complete:
                if db.cancelled(identifier):
                    raise InterruptedError("Cancelled before page analysis; model/view checkpoints preserved")
                db.event(identifier, f"{name}: selecting polygons and evaluating boundaries")
                outputs, evidence = combine(candidates, cfg)
                # GT is read only after selection; never available to fusion.
                if has_gt:
                    truth = datasets.truth(cfg.dataset_id, name)
                    for method, rows in outputs.items():
                        if db.cancelled(identifier):
                            raise InterruptedError("Cancelled during evaluation; model/view checkpoints preserved")
                        metrics[method] = measure(rows, truth, size, cfg.boundary_tolerance)
                    for key, rows in view_rows.items():
                        if db.cancelled(identifier):
                            raise InterruptedError("Cancelled during evaluation; model/view checkpoints preserved")
                        view_metrics[key] = measure(rows, truth, size, cfg.boundary_tolerance)
                for provider in cfg.providers:
                    clean = view_rows[f"{provider}|{cfg.views[0].key}"]
                    for view in cfg.views[1:]:
                        rows = view_rows[f"{provider}|{view.key}"]
                        matrix = overlap_matrix(clean, rows)
                        best = matrix.max(axis=1) if len(rows) else np.zeros(len(clean))
                        stability[f"{provider}|{view.key}"] = {
                            "clean_recovery": float(np.mean(best >= cfg.group_iou)) if len(best) else 0.,
                            "mean_best_iou": float(np.mean(best)) if len(best) else 0.,
                            "latency": timings[f"{provider}|{view.key}"]}
            payload = {"image": name, "size": size, "image_sha256": digest, "has_gt": has_gt,
                       "complete": complete, "errors": errors, "metrics": metrics, "view_metrics": view_metrics,
                       "stability": stability, "timings": timings, "evidence": evidence,
                       "outputs": {k: [a.model_dump(exclude={"mask_rle"}) for a in rows] for k, rows in outputs.items()}}
            db.save_page(identifier, name, payload, "done" if complete else "partial")
            metric_pages[name] = {k: payload[k] for k in ("image", "size", "has_gt", "complete", "metrics", "view_metrics", "stability")}
            db.update(identifier, report=live_summary(list(metric_pages.values())))
            consecutive_errors = 0 if complete else consecutive_errors + 1
            if consecutive_errors >= 3:
                raise RuntimeError("Three consecutive incomplete pages; stopped to avoid wasting GPU time")
        if db.cancelled(identifier):
            raise InterruptedError("Cancelled before final analysis")
        db.event(identifier, "Generating paired statistics, CSV and figures")
        report = export_analysis(directory, list(metric_pages.values()), cfg.seed, cfg.bootstrap_samples)
        if db.cancelled(identifier):
            raise InterruptedError("Cancelled after analysis; artifacts preserved")
        db.update(identifier, report=report, status="partial" if report["incomplete_pages"] else "done")
        db.event(identifier, "Experiment finished; results and figures saved")
    except InterruptedError as error:
        db.update(identifier, status="cancelled", message=str(error))
        db.event(identifier, str(error))
    except Exception as error:
        db.update(identifier, status="error", message=f"{type(error).__name__}: {error}")
        db.event(identifier, f"ERROR {type(error).__name__}: {error}")
        raise
    finally:
        stop_heartbeat.set()
        ticker.join(timeout=5)
        pool.close()
        db.update(identifier, elapsed=previous_elapsed+time.monotonic()-started, heartbeat=time.time(), pid=None)
        lock.close()
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--id", required=True)
    args = parser.parse_args()
    run(args.db, args.id)


if __name__ == "__main__":
    main()
