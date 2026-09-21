"""Reuse loaded networks across views and pages, including isolated environments."""

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from labelbench.contracts import Annotation
from labelbench.inference_options import ProviderOptions, current_options
from labelbench.registry import default_registry
from labelbench.robust_config import RobustConfig
from labelbench.settings import Settings


class InferencePool:
    def __init__(self, settings: Settings, config: RobustConfig, log_dir: Path) -> None:
        self.settings, self.config = settings, config
        self.providers = default_registry(settings)
        self.children: dict[str, subprocess.Popen] = {}
        self.logs = []
        self.log_dir = log_dir
        self.reader = ThreadPoolExecutor(max_workers=1)

    def predict(self, name: str, path: Path) -> list[Annotation]:
        isolated = {"docufcn_onnx": ("docufcn", self.settings.dla_python,
                                     self.settings.models_dir / "onnx" / "docufcn.onnx"),
                    "eynollah_textline": ("eynollah", self.settings.eynollah_python,
                                           self.settings.eynollah_checkpoint)}
        if name in isolated:
            if name not in self.children:
                kind, python, checkpoint = isolated[name]
                log = (self.log_dir / f"{name}.log").open("a", encoding="utf-8")
                self.logs.append(log)
                self.children[name] = subprocess.Popen(
                    [str(python), "scripts/robust_isolated_worker.py", kind, str(checkpoint)],
                    cwd=self.settings.root_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log,
                    text=True, encoding="utf-8", bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            child = self.children[name]
            child.stdin.write(json.dumps({"image": str(path)}) + "\n")
            child.stdin.flush()
            try:
                response = self.reader.submit(child.stdout.readline).result(timeout=300)
            except TimeoutError:
                child.kill()
                child.wait()
                self.children.pop(name)
                raise RuntimeError(f"{name}: inference timeout") from None
            if not response:
                self.children.pop(name)
                raise RuntimeError(f"{name}: worker exited; see worker log")
            data = json.loads(response)
            if "error" in data:
                raise RuntimeError(data["error"])
            rows = [Annotation.model_validate({**row, "provider": name}) for row in data["annotations"]]
        else:
            token = current_options.set(ProviderOptions(confidence=self.config.confidence.get(name)))
            try:
                rows = self.providers[name].annotate(path).annotations
            finally:
                current_options.reset(token)
        threshold = self.config.confidence.get(name, 0.)
        return [a for a in rows if a.label in {"text", "text_line"} and a.score >= threshold]

    def close(self) -> None:
        for child in self.children.values():
            if child.poll() is None:
                child.stdin.close()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        self.reader.shutdown(wait=False, cancel_futures=True)
        for log in self.logs:
            log.close()
