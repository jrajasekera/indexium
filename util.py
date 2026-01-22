from __future__ import annotations

import hashlib
import os


def get_file_hash(filepath: str, block_size: int = 65536) -> str | None:
    """Calculates the SHA256 hash of a file to uniquely identify it.
    Uses the first 10 blocks and last 10 blocks for a better representation
    while maintaining performance."""
    sha256 = hashlib.sha256()
    try:
        file_size = os.path.getsize(filepath)
        with open(filepath, "rb") as f:
            # Read first 10 blocks
            for _i in range(10):
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
                for _i in range(min(10, remaining_blocks)):
                    data = f.read(block_size)
                    if not data:  # Should not happen but just in case
                        break
                    sha256.update(data)

        return sha256.hexdigest()
    except OSError:
        print(f"  - [Hash Error] Could not read file: {filepath}")
        return None
    except Exception as e:
        print(f"  - [Hash Error] Error processing file {filepath}: {str(e)}")
        return None
