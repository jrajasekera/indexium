import hashlib
import os

def get_file_hash(filepath, block_size=65536):
    """Calculates the SHA256 hash of a file to uniquely identify it.
    Uses the first 25 blocks and last 25 blocks for a better representation
    while maintaining performance."""
    sha256 = hashlib.sha256()
    try:
        file_size = os.path.getsize(filepath)
        with open(filepath, 'rb') as f:
            # Read first 25 blocks
            for i in range(25):
                data = f.read(block_size)
                if not data:  # EOF reached
                    break
                sha256.update(data)

            # If file is large enough, position for last 25 blocks
            remaining_blocks = (file_size - (25 * block_size)) // block_size
            if remaining_blocks > 0:
                # Seek to the position where the last 25 blocks begin
                f.seek(max(25 * block_size, file_size - (25 * block_size)))

                # Read the last 25 blocks (or what remains)
                for i in range(min(25, remaining_blocks)):
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
