"""Set manual_review_status for scanned videos.

Defaults to setting all videos to pending so they appear in the manual review UI.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections.abc import Iterable

from config import Config

VALID_STATUSES = {"pending", "in_progress", "done", "no_people", "not_required"}
VALID_SCOPES = {"all", "not_required"}


def _fetch_counts(conn: sqlite3.Connection) -> list[tuple[str, int]]:
    rows = conn.execute(
        """
        SELECT manual_review_status, COUNT(*)
        FROM scanned_files
        GROUP BY manual_review_status
        ORDER BY manual_review_status
        """
    ).fetchall()
    return [(row[0], int(row[1])) for row in rows]


def _print_counts(label: str, counts: Iterable[tuple[str, int]]) -> None:
    print(label)
    for status, count in counts:
        print(f"  {status}: {count}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Update manual_review_status for scanned videos.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to the SQLite DB (defaults to INDEXIUM_DB or video_faces.db).",
    )
    parser.add_argument(
        "--status",
        default="pending",
        help="Target status (pending, in_progress, done, no_people, not_required).",
    )
    parser.add_argument(
        "--scope",
        default="all",
        help="Update scope: all or not_required.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    status = str(args.status).strip()
    scope = str(args.scope).strip()

    if status not in VALID_STATUSES:
        raise SystemExit(
            f"Invalid --status '{status}'. Valid: {', '.join(sorted(VALID_STATUSES))}."
        )
    if scope not in VALID_SCOPES:
        raise SystemExit(f"Invalid --scope '{scope}'. Valid: {', '.join(sorted(VALID_SCOPES))}.")

    db_path = args.db or Config().DATABASE_FILE

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        _print_counts("Before:", _fetch_counts(conn))

        if scope == "not_required":
            cursor = conn.execute(
                """
                UPDATE scanned_files
                SET manual_review_status = ?
                WHERE manual_review_status = 'not_required'
                """,
                (status,),
            )
        else:
            cursor = conn.execute(
                "UPDATE scanned_files SET manual_review_status = ?",
                (status,),
            )

        conn.commit()
        print(f"Updated rows: {cursor.rowcount}")
        _print_counts("After:", _fetch_counts(conn))


if __name__ == "__main__":
    main()
