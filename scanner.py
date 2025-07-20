import os
import sqlite3
import cv2
import face_recognition
import numpy as np
from sklearn.cluster import DBSCAN
import pickle

# --- CONFIGURATION ---
# IMPORTANT: Update these paths before running!
VIDEO_DIRECTORY = "/mnt/truenas_videos"  # The path where your SMB share is mounted
DATABASE_FILE = "video_faces.db"
# How many frames to skip between face scans. Higher is faster but less thorough.
# A value of 25 means it will process roughly one frame per second for a 25fps video.
FRAME_SKIP = 25


def setup_database():
    """Initializes the SQLite database and creates the necessary tables."""
    print("Setting up database...")
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        # Table to track which files have been scanned
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scanned_files (
                filepath TEXT PRIMARY KEY
            )
        ''')
        # Table to store face data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_filepath TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                face_location TEXT NOT NULL, -- Storing location as a string "top,right,bottom,left"
                face_encoding BLOB NOT NULL,
                cluster_id INTEGER DEFAULT NULL,
                person_name TEXT DEFAULT NULL
            )
        ''')
        conn.commit()
    print("Database setup complete.")


def is_file_scanned(filepath):
    """Checks the database to see if a video file has already been scanned."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM scanned_files WHERE filepath = ?", (filepath,))
        return cursor.fetchone() is not None


def mark_file_as_scanned(filepath):
    """Marks a file as scanned in the database."""
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO scanned_files (filepath) VALUES (?)", (filepath,))
        conn.commit()


def scan_videos():
    """
    Scans the video directory, processes each video to find faces,
    and stores their encodings in the database.
    """
    print("Starting video scan...")
    video_files = []
    for root, _, files in os.walk(VIDEO_DIRECTORY):
        for file in files:
            # Add more video extensions if needed
            if file.lower().endswith(('.mp4', '.mkv', '.mov', '.avi')):
                video_files.append(os.path.join(root, file))

    print(f"Found {len(video_files)} video files.")

    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        for i, video_path in enumerate(video_files):
            if is_file_scanned(video_path):
                print(f"({i + 1}/{len(video_files)}) Skipping already scanned file: {video_path}")
                continue

            print(f"({i + 1}/{len(video_files)}) Processing: {video_path}")
            try:
                video_capture = cv2.VideoCapture(video_path)
                if not video_capture.isOpened():
                    print(f"  - Error: Could not open video file.")
                    continue

                frame_count = 0
                faces_found_in_video = 0
                while True:
                    ret, frame = video_capture.read()
                    if not ret:
                        break  # End of video

                    # Only process every Nth frame
                    if frame_count % FRAME_SKIP == 0:
                        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                        # Find all the faces and their encodings in the current frame
                        face_locations = face_recognition.face_locations(rgb_frame)
                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                        for location, encoding in zip(face_locations, face_encodings):
                            # Serialize data for storage
                            location_str = ",".join(map(str, location))
                            encoding_blob = pickle.dumps(encoding)

                            cursor.execute('''
                                INSERT INTO faces (video_filepath, frame_number, face_location, face_encoding)
                                VALUES (?, ?, ?, ?)
                            ''', (video_path, frame_count, location_str, encoding_blob))
                            faces_found_in_video += 1

                    frame_count += 1

                video_capture.release()
                conn.commit()
                mark_file_as_scanned(video_path)
                print(f"  - Finished. Found {faces_found_in_video} faces.")

            except Exception as e:
                print(f"  - An error occurred while processing {video_path}: {e}")

    print("Video scanning complete.")


def cluster_faces():
    """
    Fetches all face encodings from the database and uses DBSCAN to group them.
    Updates the database with the cluster ID for each face.
    """
    print("Starting face clustering...")
    with sqlite3.connect(DATABASE_FILE) as conn:
        cursor = conn.cursor()
        # Fetch only faces that haven't been clustered yet
        cursor.execute("SELECT id, face_encoding FROM faces WHERE cluster_id IS NULL")
        rows = cursor.fetchall()

        if not rows:
            print("No new faces to cluster.")
            return

        print(f"Found {len(rows)} new faces to cluster.")

        face_ids = [row[0] for row in rows]
        # Deserialize the encodings
        encodings = [pickle.loads(row[1]) for row in rows]

        # Use DBSCAN to find clusters of faces
        # The `eps` parameter is the most important setting. It's the maximum distance
        # between two samples for one to be considered as in the neighborhood of the other.
        # This may require some tuning based on your specific videos. 0.4 is a good start.
        # `min_samples` is the number of samples in a neighborhood for a point to be considered as a core point.
        clt = DBSCAN(metric="euclidean", n_jobs=-1, eps=0.4, min_samples=5)
        clt.fit(encodings)

        # Get the highest existing cluster_id to not reuse IDs
        cursor.execute("SELECT MAX(cluster_id) FROM faces")
        max_cluster_id = cursor.fetchone()[0]
        if max_cluster_id is None:
            max_cluster_id = 0

        label_offset = max_cluster_id + 1

        print(f"Clustering complete. Found {len(np.unique(clt.labels_))} unique groups (including noise).")

        # Update the database with the new cluster IDs
        for face_id, label in zip(face_ids, clt.labels_):
            # We label noise points (-1) as NULL so we can potentially re-cluster them later
            if label == -1:
                cluster_id = None
            else:
                cluster_id = int(label) + label_offset

            cursor.execute("UPDATE faces SET cluster_id = ? WHERE id = ?", (cluster_id, face_id))

        conn.commit()
    print("Face clustering complete.")


if __name__ == "__main__":
    setup_database()
    scan_videos()
    cluster_faces()
