"""Migrate NFO actors to source='indexium' when they match DB people."""

from __future__ import annotations

import argparse
import os
import sqlite3
from collections.abc import Iterable

from nfo_services import NfoParseError, NfoService


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert matching NFO actors to source='indexium'.",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Path to the SQLite DB (defaults to INDEXIUM_DB or video_faces.db).",
    )
    parser.add_argument(
        "--file-hash",
        action="append",
        default=[],
        help="Specific file hash to migrate (repeatable).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all scanned files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing.",
    )
    return parser.parse_args()


def _get_db_people(conn: sqlite3.Connection, file_hash: str) -> list[str]:
    people: set[str] = set()

    rows = conn.execute(
        "SELECT DISTINCT person_name FROM faces WHERE file_hash = ? AND person_name IS NOT NULL",
        (file_hash,),
    ).fetchall()
    for row in rows:
        people.add(row[0])

    rows = conn.execute(
        "SELECT DISTINCT person_name FROM video_people WHERE file_hash = ?",
        (file_hash,),
    ).fetchall()
    for row in rows:
        people.add(row[0])

    return sorted(people, key=str.lower)


def _format_list(items: Iterable[str]) -> str:
    return ", ".join(items)


def main() -> None:
    args = _parse_args()

    if not args.all and not args.file_hash:
        raise SystemExit("Provide --all or at least one --file-hash.")

    db_path = args.db or os.environ.get("INDEXIUM_DB", "video_faces.db")
    nfo_service = NfoService()

    total = 0
    skipped_missing = 0
    skipped_no_people = 0
    skipped_no_nfo = 0
    skipped_no_conflict = 0
    parse_errors = 0
    updated_files = 0
    updated_actors = 0
    removed_duplicates = 0

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if args.file_hash:
            file_hashes = list(dict.fromkeys(args.file_hash))
        else:
            rows = conn.execute(
                "SELECT file_hash FROM scanned_files WHERE last_known_filepath IS NOT NULL"
            ).fetchall()
            file_hashes = [row[0] for row in rows]

        for file_hash in file_hashes:
            total += 1
            row = conn.execute(
                "SELECT last_known_filepath FROM scanned_files WHERE file_hash = ?",
                (file_hash,),
            ).fetchone()
            if not row:
                skipped_missing += 1
                continue

            video_path = row["last_known_filepath"]
            if not video_path:
                skipped_missing += 1
                continue

            db_people = _get_db_people(conn, file_hash)
            if not db_people:
                skipped_no_people += 1
                continue

            nfo_path = nfo_service.find_nfo_path(video_path)
            if not nfo_path:
                skipped_no_nfo += 1
                continue

            try:
                actors = nfo_service.read_actors(nfo_path)
            except NfoParseError:
                parse_errors += 1
                continue

            db_names = {name.lower() for name in db_people}
            has_conflict = any(
                actor.source != "indexium" and actor.name.lower() in db_names for actor in actors
            )
            if not has_conflict:
                skipped_no_conflict += 1
                continue

            if args.dry_run:
                print(
                    f"Would update {os.path.basename(video_path)}: {_format_list(db_people)}",
                )
                continue

            updated, removed = nfo_service.migrate_actor_sources(nfo_path, db_people)
            if updated or removed:
                updated_files += 1
                updated_actors += updated
                removed_duplicates += removed

    print("Migration summary:")
    print(f"  Files considered: {total}")
    print(f"  Updated files: {updated_files}")
    print(f"  Actors updated: {updated_actors}")
    print(f"  Duplicates removed: {removed_duplicates}")
    print(f"  Skipped missing files: {skipped_missing}")
    print(f"  Skipped (no people): {skipped_no_people}")
    print(f"  Skipped (no NFO): {skipped_no_nfo}")
    print(f"  Skipped (no conflicts): {skipped_no_conflict}")
    print(f"  Parse errors: {parse_errors}")


if __name__ == "__main__":
    main()
