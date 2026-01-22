from __future__ import annotations

from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch

from util import get_file_hash


def test_get_file_hash_consistent(tmp_path: Path) -> None:
    f = tmp_path / "test.txt"
    f.write_text("hello")
    h1 = get_file_hash(str(f))
    h2 = get_file_hash(str(f))
    assert h1 == h2


def test_get_file_hash_changes(tmp_path: Path) -> None:
    f = tmp_path / "file.txt"
    f.write_text("a")
    h1 = get_file_hash(str(f))
    f.write_text("b")
    h2 = get_file_hash(str(f))
    assert h1 != h2


def test_get_file_hash_missing(tmp_path: Path) -> None:
    missing = tmp_path / "missing.txt"
    assert get_file_hash(str(missing)) is None


def test_get_file_hash_large_file(tmp_path: Path) -> None:
    # Create a file larger than 10 blocks to exercise the tail-reading logic
    big_file = tmp_path / "big.bin"
    big_file.write_bytes(b"a" * (65536 * 11))  # 11 blocks
    hash_value = get_file_hash(str(big_file))
    assert isinstance(hash_value, str) and len(hash_value) == 64


def test_get_file_hash_generic_exception(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    # Create a real file so os.path.getsize succeeds
    f = tmp_path / "file.txt"
    f.write_text("data")

    def bad_open(*args: object, **kwargs: object) -> None:
        raise ValueError("boom")

    monkeypatch.setattr("builtins.open", bad_open)
    assert get_file_hash(str(f)) is None


def test_get_file_hash_tail_break(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    import os

    # File with exactly 10 blocks
    ten_block_file = tmp_path / "ten.bin"
    ten_block_file.write_bytes(b"a" * (65536 * 10))

    # Pretend the file is larger so the tail-reading loop runs
    monkeypatch.setattr(os.path, "getsize", lambda _: 65536 * 11)

    hash_value = get_file_hash(str(ten_block_file))
    assert isinstance(hash_value, str) and len(hash_value) == 64
