"""Convert editor database to a separate, registered polygon dataset."""

import argparse
import json
from pathlib import Path

from labelbench.archive_conversion import convert_archive


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--output", type=Path, default=Path("data/output"))
    args = parser.parse_args()
    print(json.dumps(convert_archive(args.root, args.destination, args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
