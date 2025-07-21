# Indexium

A Python-based video face scanning and tagging application that automatically detects faces in videos, groups them by person using machine learning clustering, and allows you to manually tag and organize people found in your video collection.

## Features

- **Automatic Face Detection**: Scans video files and extracts faces using computer vision
- **Smart Clustering**: Groups similar faces together using DBSCAN clustering algorithm
- **Web-based Tagging Interface**: Clean, responsive web UI for reviewing and naming face groups
- **Parallel Processing**: Multi-core video processing for faster scanning
- **File Hash-based Tracking**: Tracks videos by content hash, handles moved/renamed files gracefully
- **Metadata Writing**: Embeds person tags directly into video file metadata
- **Face Group Management**: Split groups, merge people, rename, and organize your tags
- **Progress Tracking**: Visual progress indicators and statistics

## Screenshots

The application provides an intuitive web interface for:
- Reviewing untagged face groups
- Naming individuals found in videos
- Managing and editing existing tags
- Viewing all identified people in your collection

## Installation

### Prerequisites

- Python 3.10 or higher
- OpenCV dependencies (for video processing)
- dlib dependencies (for face recognition)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd indexium
```

2. Install dependencies using uv (recommended):
```bash
uv sync
```

Or using pip:
```bash
pip install -e .
```

### System Dependencies

On Ubuntu/Debian:
```bash
sudo apt update
sudo apt install cmake libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev python3-dev
```

On macOS:
```bash
brew install cmake
```

## Usage

### 1. Set Environment Variable

Set the path to your video directory:
```bash
export INDEXIUM_VIDEO_DIR="/path/to/your/videos"
```

### 2. Scan Videos

Run the scanner to detect and cluster faces:
```bash
python scanner.py
```

The scanner will:
- Process all video files in your directory (MP4, MKV, MOV, AVI)
- Extract faces from video frames
- Group similar faces using machine learning
- Save results to a SQLite database

### 3. Tag Faces

Start the web application:
```bash
python app.py
```

Then open your browser to `http://localhost:5001` to:
- Review face groups and assign names
- Split incorrectly grouped faces
- Merge faces of the same person
- Manage your tagged people

### 4. Write Metadata

Once you've tagged faces, write the tags to your video files:
- Click "Write All Named Tags" in the web interface
- Tags are embedded in the video metadata as "People: Name1, Name2, ..."
- Original files are safely replaced with tagged versions

## Configuration

All configuration is centralized in `config.py`. Values are loaded from
environment variables with sensible defaults:

- `INDEXIUM_VIDEO_DIR`: directory of videos to scan (default: `test_videos`)
- `INDEXIUM_DB`: path to the SQLite database (default: `video_faces.db`)
- `FRAME_SKIP`: how many frames to skip between scans (default: 25)
- `CPU_CORES`: number of CPU cores to use (`None` uses all cores)
- `SAVE_CHUNK_SIZE`: how often to save progress (default: 10)
- `SECRET_KEY`: Flask secret key
- `FLASK_DEBUG`: run the web UI in debug mode
- `AUTO_CLASSIFY_THRESHOLD`: distance threshold for automatic face naming (default: 0.3)

## Database

The application uses SQLite with two main tables:
- `scanned_files`: Tracks processed videos by hash
- `faces`: Stores face data, locations, encodings, and tags

Database file: `video_faces.db`

## File Structure

```
indexium/
├── app.py              # Flask web application
├── scanner.py          # Video scanning and face clustering
├── util.py             # Utility functions (file hashing)
├── pyproject.toml      # Project dependencies
├── video_faces.db      # SQLite database (created on first run)
└── README.md           # This file
```

## How It Works

1. **Scanning**: The scanner processes videos frame-by-frame, detecting faces using the `face_recognition` library
2. **Encoding**: Each face is converted to a 128-dimensional encoding vector
3. **Clustering**: DBSCAN algorithm groups similar face encodings together
4. **Tagging**: Web interface allows manual review and naming of face groups
5. **Metadata**: Tags are written to video files using ffmpeg

## Technical Details

- **Face Detection**: Uses HOG-based face detection from dlib
- **Face Recognition**: 128-dimensional face encodings for comparison
- **Clustering**: DBSCAN with euclidean distance (eps=0.4, min_samples=5)
- **Parallel Processing**: Multiprocessing for video scanning
- **File Tracking**: SHA256 hashing (first and last 25 blocks) for file identification
- **Web Framework**: Flask with Tailwind CSS styling

## Troubleshooting

### Common Issues

1. **"INDEXIUM_VIDEO_DIR environment variable not set"**
   - Set the environment variable pointing to your video directory

2. **Videos not being processed**
   - Check file permissions and supported formats (MP4, MKV, MOV, AVI)
   - Ensure videos are not corrupted

3. **Face detection not working well**
   - Adjust `FRAME_SKIP` for more thorough scanning
   - Consider video quality and lighting conditions

4. **Performance issues**
   - Reduce `CPU_CORES_TO_USE` if system becomes unresponsive
   - Increase `FRAME_SKIP` for faster but less thorough scanning

### Database Issues

If you encounter database corruption:
1. Stop all processes
2. Delete `video_faces.db`
3. Restart scanning process

## Safety Features

- **Non-destructive**: Original video files are safely preserved during metadata writing
- **Graceful shutdown**: Ctrl+C during scanning saves progress before exit
- **File integrity**: Uses content hashing to avoid reprocessing moved files
- **Backup-friendly**: All data stored in single SQLite database file

## License

[TODO]

## Contributing

[TODO]

## Support

[TODO]

## Running Tests

The project includes a small pytest suite. To run it:

```bash
pytest -q
```
