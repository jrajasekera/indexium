"""Print the effective Indexium configuration used by the local app."""

from __future__ import annotations

from config import Config


def main() -> None:
    config = Config()
    print(f"DB: {config.DATABASE_FILE}")
    print(f"VIDEO_DIR: {config.VIDEO_DIR}")


if __name__ == "__main__":
    main()
