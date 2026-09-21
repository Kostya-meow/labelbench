"""Launch and supervise durable workers without tying jobs to HTTP requests."""

import hashlib
import os
import subprocess
import time
import uuid
from pathlib import Path

from labelbench.contracts import safe_child_path
from labelbench.datasets import DatasetStore
from labelbench.registry import default_registry
from labelbench.robust_config import RobustConfig
from labelbench.robust_db import RobustDB
from labelbench.robust_provenance import gpu_info, model_signatures, source_signature
from labelbench.settings import Settings
from labelbench.storage import write_json


class RobustManager:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.directory = settings.output_dir / "robustness"
        self.db = RobustDB(self.directory / "experiments.sqlite")

    def active(self) -> bool:
        return any(r["status"] in {"queued", "running", "cancelling"} for r in self.list())

    def list(self) -> list[dict]:
        rows = self.db.list()
        for row in rows:
            if row["status"] in {"running", "cancelling", "queued"} and time.time()-max(row["heartbeat"], row["updated"]) > 120:
                self.db.update(row["id"], status="interrupted", message="Heartbeat lost; resume from checkpoints")
                row["status"] = "interrupted"
        return rows

    def start(self, cfg: RobustConfig) -> dict:
        registry = default_registry(self.settings)
        for provider in cfg.providers:
            if provider not in registry:
                raise ValueError(f"Unknown provider: {provider}")
            status = registry[provider].availability()
            if not status.available:
                raise ValueError(f"{provider}: {status.detail}")
        manifest = DatasetStore(self.settings.output_dir).get(cfg.dataset_id)
        root = Path(manifest["root"])
        gt_names = {row["name"] for row in manifest["images"]}
        names = sorted(p.name for p in root.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) if cfg.include_unlabelled else sorted(gt_names)
        if cfg.images is not None:
            if set(cfg.images) - set(names):
                raise ValueError("Unknown requested image")
            names = list(dict.fromkeys(cfg.images))
        if not names:
            raise ValueError("No images")
        hashes = {n: hashlib.sha256(safe_child_path(root, n).read_bytes()).hexdigest() for n in names}
        python = self.settings.root_dir / ".venv-gpu" / "Scripts" / "python.exe"
        freeze = subprocess.run([str(python), "-m", "pip", "freeze"], capture_output=True, text=True, check=False)
        # uv-only environments may not ship pip.
        if freeze.returncode:
            freeze = subprocess.run(["uv", "pip", "freeze", "--python", str(python)], capture_output=True, text=True, check=True)
        environment = {"source_sha256": source_signature(self.settings.root_dir), "packages": freeze.stdout.splitlines(),
                       "model_sha256": model_signatures(self.settings, cfg.providers), "gpu": gpu_info(),
                       "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=self.settings.root_dir, text=True).strip(),
                       "device": "cuda", "python": str(python), "protocol": "polygon-evidence-v1"}
        identifier = uuid.uuid4().hex[:12]
        frozen_manifest = {"root": str(root), "images": names, "gt_images": sorted(gt_names & set(names)),
                           "dataset_sha256": manifest["annotation_sha256"], "image_sha256": hashes}
        self.db.create(identifier, cfg.model_dump(), frozen_manifest, environment)
        directory = self.directory / identifier
        directory.mkdir()
        write_json(directory / "protocol.json", {"config": cfg.model_dump(), "manifest": frozen_manifest, "environment": environment})
        self._launch(identifier)
        return self.db.get(identifier)

    def _launch(self, identifier: str) -> None:
        directory = self.directory / identifier
        env = os.environ.copy()
        env.update(PYTHONHASHSEED=str(self.db.get(identifier)["config"]["seed"]),
                   PYTHONIOENCODING="utf-8", LABELBENCH_DEVICE="cuda",
                   HF_HOME=str(self.settings.models_dir / "huggingface"),
                   RF_HOME=str(self.settings.models_dir / "rfdetr"))
        with (directory / "worker.log").open("a", encoding="utf-8") as log:
            child = subprocess.Popen(
                [str(self.settings.root_dir / ".venv-gpu" / "Scripts" / "python.exe"),
                 "-m", "labelbench.robust_worker", "--db", str(self.db.path), "--id", identifier],
                cwd=self.settings.root_dir, env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
                close_fds=True, creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW) if os.name == "nt" else 0,
                start_new_session=os.name != "nt")
        self.db.update(identifier, pid=child.pid)

    def cancel(self, identifier: str) -> None:
        item = self.db.get(identifier)
        if item["status"] in {"queued", "running", "cancelling"}:
            self.db.update(identifier, cancel=1, status="cancelling")
            self.db.event(identifier, "Cancel requested; finishing the current inference and preserving checkpoints")

    def resume(self, identifier: str) -> dict:
        if self.active():
            raise ValueError("Another experiment is active")
        item = self.db.get(identifier)
        if item["status"] == "done":
            raise ValueError("Experiment is already complete")
        if item["environment"]["source_sha256"] != source_signature(self.settings.root_dir):
            raise ValueError("Executable protocol changed; create a new run rather than mix versions")
        if item["environment"].get("model_sha256") != model_signatures(self.settings, item["config"]["providers"]):
            raise ValueError("Model files changed; create a new run")
        self.db.claim_resume(identifier)
        self._launch(identifier)
        return self.db.get(identifier)
