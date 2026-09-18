"""Resume a large download with several parallel HTTP range requests."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Segment:
    index: int
    start: int
    end: int
    path: Path

    @property
    def size(self) -> int:
        return self.end - self.start + 1


def build_segments(base: int, total: int, count: int, parts_dir: Path) -> list[Segment]:
    remaining = total - base
    chunk_size = (remaining + count - 1) // count
    segments: list[Segment] = []
    for index in range(count):
        start = base + index * chunk_size
        if start >= total:
            break
        end = min(total - 1, start + chunk_size - 1)
        segments.append(Segment(index, start, end, parts_dir / f"part-{index:02d}"))
    return segments


def download_segment(url: str, segment: Segment) -> None:
    if segment.path.is_file() and segment.path.stat().st_size == segment.size:
        print(f"segment {segment.index + 1}: cached")
        return
    segment.path.unlink(missing_ok=True)
    subprocess.run(
        [
            "curl.exe",
            "-L",
            "--fail",
            "--retry",
            "10",
            "--retry-all-errors",
            "--silent",
            "--show-error",
            "--range",
            f"{segment.start}-{segment.end}",
            "-o",
            str(segment.path),
            url,
        ],
        check=True,
    )
    actual_size = segment.path.stat().st_size
    if actual_size != segment.size:
        raise RuntimeError(
            f"Segment {segment.index + 1} has {actual_size} bytes, expected {segment.size}"
        )
    print(f"segment {segment.index + 1}: complete ({actual_size / 1024 / 1024:.1f} MiB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("target", type=Path)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--connections", type=int, default=8)
    arguments = parser.parse_args()

    target = arguments.target.resolve()
    partial = target.with_name(f"{target.name}.part")
    parts_dir = target.with_name(f"{target.name}.parts")
    manifest_path = parts_dir / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.is_file() and target.stat().st_size == arguments.size:
        print(f"Already downloaded: {target}")
        return
    if target.exists():
        raise RuntimeError(f"Invalid completed file size: {target.stat().st_size}")

    parts_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        base = int(manifest["base"])
        if manifest["url"] != arguments.url or int(manifest["size"]) != arguments.size:
            raise RuntimeError("Download manifest does not match the requested file")
    else:
        base = partial.stat().st_size if partial.is_file() else 0
        manifest_path.write_text(
            json.dumps({"base": base, "size": arguments.size, "url": arguments.url}),
            encoding="utf-8",
        )
    if not partial.exists():
        partial.touch()
    if partial.stat().st_size != base:
        raise RuntimeError("Partial file changed after segmented download started")

    segments = build_segments(base, arguments.size, arguments.connections, parts_dir)
    print(f"Downloading {arguments.size - base} remaining bytes in {len(segments)} segments...")
    with ThreadPoolExecutor(max_workers=arguments.connections) as executor:
        list(executor.map(lambda segment: download_segment(arguments.url, segment), segments))

    with partial.open("ab") as destination:
        for segment in segments:
            with segment.path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    if partial.stat().st_size != arguments.size:
        raise RuntimeError(f"Final size is {partial.stat().st_size}, expected {arguments.size}")
    partial.replace(target)
    shutil.rmtree(parts_dir)
    print(f"Download complete: {target}")


if __name__ == "__main__":
    main()
