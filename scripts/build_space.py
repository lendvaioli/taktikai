"""Build the static Hugging Face Space into space/ (see configs/space.yaml)."""

import sys

from match_tagger.cli import use_headless_matplotlib


def main() -> int:
    """Entry point: python scripts/build_space.py."""
    use_headless_matplotlib()
    from match_tagger.space.build import build

    print(f"Built {build()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
