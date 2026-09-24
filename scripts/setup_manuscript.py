"""Install the pinned branch from a local editable checkout into its own environment."""

import subprocess
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    repository = root / "data/vendor/manuscript-ocr"
    revision = "1bc3b501d3f3124cc166613b279c463dadc3ef6d"
    if not repository.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--single-branch", "--branch", "v_0_1_13",
                        "https://github.com/konstantinkozhin/manuscript-ocr.git", str(repository)], check=True)
    current = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True).strip()
    if current != revision:
        raise RuntimeError(f"Checkout is {current}, expected tested revision {revision}; no local files were reset")
    python = root / ".venv-manuscript/Scripts/python.exe"
    if not python.is_file():
        subprocess.run(["uv", "venv", str(python.parents[1]), "--python", "3.10"], check=True)
    subprocess.run(["uv", "pip", "install", "--python", str(python), "--no-deps", "-e", str(repository)], check=True)
    subprocess.run(["uv", "pip", "install", "--python", str(python), "-r",
                    str(root / "scripts/requirements-manuscript.txt")], check=True)
    subprocess.run([str(python), str(root / "scripts/manuscript_worker.py"), "--prefetch", "--device", "cuda"], check=True)
    subprocess.run([str(python), str(root / "scripts/optimize_manuscript_onnx.py")], check=True)
    images = [root / "архив 2/109.jpg", root / "архив 2/1090.jpg"]
    if not all(image.is_file() for image in images):
        images = [repository / "example/images/crop1.png", repository / "example/images/crop2.png"]
    subprocess.run([str(python), str(root / "scripts/verify_manuscript_onnx.py"),
                    "--images", *map(str, images)], check=True)


if __name__ == "__main__":
    main()
