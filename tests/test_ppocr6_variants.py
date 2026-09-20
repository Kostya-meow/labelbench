from labelbench.registry import default_registry
from labelbench.settings import Settings


def test_three_variants_have_separate_models_and_cache_keys(tmp_path, monkeypatch) -> None:
    from PIL import Image

    from labelbench.annotation_cache import cache_key

    monkeypatch.setenv('LABELBENCH_ROOT', str(tmp_path))
    providers = default_registry(Settings.from_environment())
    image = tmp_path / 'page.png'
    Image.new('RGB', (16, 16)).save(image)
    keys = set()
    for name, variant in [('ppocr6', 'medium'), ('ppocr6_small', 'small'), ('ppocr6_tiny', 'tiny')]:
        provider = providers[name]
        assert provider.name == name
        assert provider.model_name == f'PaddlePaddle/PP-OCRv6_{variant}_det_safetensors'
        keys.add(cache_key(image, provider))
    assert len(keys) == 3
