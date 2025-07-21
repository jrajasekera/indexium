from util import get_file_hash

def test_get_file_hash_consistent(tmp_path):
    f = tmp_path / 'test.txt'
    f.write_text('hello')
    h1 = get_file_hash(str(f))
    h2 = get_file_hash(str(f))
    assert h1 == h2


def test_get_file_hash_changes(tmp_path):
    f = tmp_path / 'file.txt'
    f.write_text('a')
    h1 = get_file_hash(str(f))
    f.write_text('b')
    h2 = get_file_hash(str(f))
    assert h1 != h2


def test_get_file_hash_missing(tmp_path):
    missing = tmp_path / 'missing.txt'
    assert get_file_hash(str(missing)) is None
