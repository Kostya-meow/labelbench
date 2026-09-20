from pathlib import Path

from PIL import Image

from labelbench.contracts import ProviderResult
from labelbench.providers.base import AnnotationProvider, ProviderAvailability
from labelbench.service import AnnotationService
from labelbench.settings import Settings


class FakeProvider(AnnotationProvider):
    name = "fake"
    model_name = "fake-v1"

    def __init__(self) -> None:
        self.calls = 0
        self.available = True
        self._points_per_side = 24

    def availability(self) -> ProviderAvailability:
        return ProviderAvailability(self.available, "test")

    def annotate(self, image_path: Path) -> ProviderResult:
        self.calls += 1
        with Image.open(image_path) as image:
            return ProviderResult(provider=self.name, model=self.model_name,
                                  image_name=image_path.name, image_size=list(image.size),
                                  annotations=[], elapsed_seconds=0.1)


def setup_service(tmp_path: Path, monkeypatch) -> tuple[AnnotationService, FakeProvider, Path]:
    monkeypatch.setenv("LABELBENCH_ROOT", str(tmp_path))
    settings = Settings.from_environment()
    settings.ensure_directories()
    path = settings.images_dir / "test.png"
    Image.new("RGB", (20, 20), "white").save(path)
    provider = FakeProvider()
    return AnnotationService(settings, {"fake": provider}), provider, path


def test_persistent_cache_and_force(tmp_path: Path, monkeypatch) -> None:
    service, provider, _ = setup_service(tmp_path, monkeypatch)
    assert not service.run("test.png", ["fake"]).providers["fake"].cached
    provider.available = False
    # Cache must remain available after a service restart and without GPU inference.
    restarted = AnnotationService(service.settings, {"fake": provider})
    assert restarted.run("test.png", ["fake"]).providers["fake"].cached
    assert provider.calls == 1
    provider.available = True
    assert not restarted.run("test.png", ["fake"], force=True).providers["fake"].cached
    assert provider.calls == 2


def test_image_and_settings_changes_invalidate_cache(tmp_path: Path, monkeypatch) -> None:
    service, provider, path = setup_service(tmp_path, monkeypatch)
    service.run("test.png", ["fake"])
    Image.new("RGB", (20, 20), "black").save(path)
    service.run("test.png", ["fake"])
    provider._points_per_side = 48
    service.run("test.png", ["fake"])
    provider.model_name = "fake-v2"
    service.run("test.png", ["fake"])
    assert provider.calls == 4


def test_corrupt_cache_recomputes(tmp_path: Path, monkeypatch) -> None:
    service, provider, _ = setup_service(tmp_path, monkeypatch)
    result = service.run("test.png", ["fake"]).providers["fake"]
    (service.settings.output_dir / "cache" / f"{result.cache_key}.json").write_text("broken")
    service.run("test.png", ["fake"])
    assert provider.calls == 2


def test_legacy_result_is_reused_and_marked(tmp_path: Path, monkeypatch) -> None:
    service, provider, path = setup_service(tmp_path, monkeypatch)
    legacy = provider.annotate(path)
    service._write_provider_result("old", legacy)
    result = service.run("test.png", ["fake"]).providers["fake"]
    assert result.cached and result.cache_origin == "legacy"
    assert provider.calls == 1
    provider._points_per_side = 48
    assert not service.run("test.png", ["fake"]).providers["fake"].cached
    assert provider.calls == 2
