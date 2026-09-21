import builtins
from pathlib import Path
from types import SimpleNamespace

from labelbench.providers.ppocr6 import PPOCR6Provider
from labelbench.settings import Settings


def test_health_does_not_initialize_lazy_transformers(monkeypatch, tmp_path: Path) -> None:
    """Health polling must not race model initialization in a worker thread."""
    from labelbench.providers import ppocr6

    package = tmp_path / "transformers"
    (package / "models" / "pp_ocrv6_medium_det").mkdir(parents=True)
    monkeypatch.setattr(ppocr6, "find_spec", lambda name: SimpleNamespace(origin=str(package / "__init__.py")))
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name == "transformers":
            raise AssertionError("Health must not import Transformers")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assert PPOCR6Provider(Settings.from_environment()).availability().available
