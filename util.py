"""Utility helpers for file operations."""

from __future__ import annotations

import hashlib
import os
from typing import Optional

def get_file_hash(filepath: str, block_size: int = 65536) -> Optional[str]:
    """Return a stable hash for ``filepath``.

    The function reads the first and last ten ``block_size`` byte chunks of the
    file to construct a SHA256 hash.  Only a subset of the file is read so that
    even very large media files can be fingerprinted quickly.  ``None`` is
    returned if the file cannot be read.

    Parameters
    ----------
    filepath:
        Absolute path of the file to hash.
    block_size:
        Size of the chunks to read from the file in bytes.

    Returns
    -------
    Optional[str]
        The hexadecimal digest of the SHA256 hash or ``None`` if the file could
        not be processed.
    """
    sha256 = hashlib.sha256()
    try:
        file_size = os.path.getsize(filepath)
        with open(filepath, 'rb') as f:
            # Read first 10 blocks
            for i in range(10):
                data = f.read(block_size)
                if not data:  # EOF reached
                    break
                sha256.update(data)

            # If file is large enough, position for last 10 blocks
            remaining_blocks = (file_size - (10 * block_size)) // block_size
            if remaining_blocks > 0:
                # Seek to the position where the last 10 blocks begin
                f.seek(max(10 * block_size, file_size - (10 * block_size)))

                # Read the last 10 blocks (or what remains)
                for i in range(min(10, remaining_blocks)):
                    data = f.read(block_size)
                    if not data:  # Should not happen but just in case
                        break
                    sha256.update(data)

        return sha256.hexdigest()
    except IOError:
        print(f"  - [Hash Error] Could not read file: {filepath}")
        return None
    except Exception as e:
        print(f"  - [Hash Error] Error processing file {filepath}: {str(e)}")
        return None
