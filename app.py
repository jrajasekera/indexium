import os
import sqlite3
import sys

import cv2
import ffmpeg
from flask import Flask, render_template_string, request, redirect, url_for, Response

# --- CONFIGURATION ---
# Get video directory from environment variable
VIDEO_DIRECTORY = os.environ.get("INDEXIUM_VIDEO_DIR")
if VIDEO_DIRECTORY is None:
    print("Error: INDEXIUM_VIDEO_DIR environment variable not set")
    print("Please set this variable to the directory containing your videos")
    sys.exit(1)

DATABASE_FILE = "video_faces.db"

app = Flask(__name__)


def get_db_connection():
    """Creates a database connection."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn


@app.route('/')
def index():
    """Main page, displays unnamed face clusters."""
    conn = get_db_connection()
    # Find clusters that are not yet named
    clusters = conn.execute('''
        SELECT DISTINCT cluster_id
        FROM faces
        WHERE cluster_id IS NOT NULL AND person_name IS NULL
    ''').fetchall()

    cluster_data = []
    for cluster in clusters:
        cluster_id = cluster['cluster_id']
        # Get a few sample faces for this cluster to display as thumbnails
        sample_faces = conn.execute('''
            SELECT id, video_filepath FROM faces WHERE cluster_id = ? LIMIT 5
        ''', (cluster_id,)).fetchall()
        cluster_data.append({'id': cluster_id, 'faces': sample_faces})

    conn.close()

    # This is a self-contained HTML template using Tailwind CSS for styling.
    # No separate HTML file is needed.
    return render_template_string('''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Video Face Tagger</title>
            <script src="https://cdn.tailwindcss.com"></script>
            <link rel="preconnect" href="https://fonts.googleapis.com">
            <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
            <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
            <style>
                body { font-family: 'Inter', sans-serif; }
            </style>
        </head>
        <body class="bg-gray-100 text-gray-800">
            <div class="container mx-auto p-4 sm:p-6 lg:p-8">
                <div class="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6">
                    <div>
                        <h1 class="text-3xl font-bold text-gray-900">Unnamed Face Groups</h1>
                        <p class="mt-1 text-gray-600">Identify the person in each group. {{ clusters|length }} groups remaining.</p>
                    </div>
                    <form action="/write_metadata" method="post" onsubmit="return confirm('This will write metadata to your video files. This is a non-destructive process that creates new files, but please ensure you have backups. Are you sure?');">
                        <button type="submit" class="mt-4 sm:mt-0 bg-green-600 text-white font-semibold py-2 px-4 rounded-lg shadow-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-75 transition-colors">
                            Write All Named Tags to Files
                        </button>
                    </form>
                </div>

                {% if not clusters %}
                    <div class="bg-white p-8 rounded-lg shadow-md text-center">
                        <h2 class="text-2xl font-semibold">All Done!</h2>
                        <p class="mt-2 text-gray-600">No unnamed face groups found. You can run the scanner again to find more faces or click the button above to write any remaining tags.</p>
                    </div>
                {% else %}
                    <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {% for cluster in clusters %}
                        <div class="bg-white rounded-lg shadow-md p-4">
                            <h2 class="font-semibold text-lg mb-3">Group #{{ cluster.id }}</h2>
                            <div class="flex flex-wrap gap-2 mb-4">
                                {% for face in cluster.faces %}
                                    <img src="{{ url_for('get_face_thumbnail', face_id=face.id) }}" alt="Face from video" class="w-20 h-20 object-cover rounded-md bg-gray-200">
                                {% endfor %}
                            </div>
                            <form action="{{ url_for('name_cluster') }}" method="post" class="flex gap-2">
                                <input type="hidden" name="cluster_id" value="{{ cluster.id }}">
                                <input type="text" name="person_name" placeholder="Enter name..." class="flex-grow p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500" required>
                                <button type="submit" class="bg-blue-600 text-white font-semibold py-2 px-4 rounded-md hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-opacity-75">Save</button>
                            </form>
                        </div>
                        {% endfor %}
                    </div>
                {% endif %}
            </div>
        </body>
        </html>
    ''', clusters=cluster_data)


@app.route('/face_thumbnail/<int:face_id>')
def get_face_thumbnail(face_id):
    """
    Dynamically generates a cropped thumbnail image for a given face ID.
    This is called by the <img> tags on the main page.
    """
    conn = get_db_connection()
    face_data = conn.execute('SELECT video_filepath, frame_number, face_location FROM faces WHERE id = ?',
                             (face_id,)).fetchone()
    conn.close()

    if not face_data:
        return "Face not found", 404

    video_path = face_data['video_filepath']
    if not os.path.exists(video_path):
        # Fallback if the video path from the DB isn't found (e.g., share unmounted)
        return f"Video file not found: {video_path}", 404

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return "Could not open video", 500

    # Go to the specific frame where the face was detected
    cap.set(cv2.CAP_PROP_POS_FRAMES, face_data['frame_number'])
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return "Could not read frame", 500

    # Unpack the face location
    top, right, bottom, left = map(int, face_data['face_location'].split(','))

    # Crop the face from the frame
    face_image = frame[top:bottom, left:right]

    # Encode the cropped image as JPEG and return it
    ret, buffer = cv2.imencode('.jpg', face_image)
    if not ret:
        return "Could not encode image", 500

    return Response(buffer.tobytes(), mimetype='image/jpeg')


@app.route('/name_cluster', methods=['POST'])
def name_cluster():
    """Handles the form submission for naming a cluster."""
    cluster_id = request.form['cluster_id']
    person_name = request.form['person_name'].strip()

    if cluster_id and person_name:
        conn = get_db_connection()
        conn.execute('UPDATE faces SET person_name = ? WHERE cluster_id = ?', (person_name, cluster_id))
        conn.commit()
        conn.close()
        print(f"Assigned name '{person_name}' to cluster {cluster_id}")

    return redirect(url_for('index'))


@app.route('/write_metadata', methods=['POST'])
def write_metadata():
    """
    Finds all videos with named faces and writes the names into the file's
    metadata 'comment' tag using ffmpeg.
    """
    print("Starting metadata write process...")
    conn = get_db_connection()
    # Find all videos that have at least one named face
    videos_to_tag = conn.execute('''
        SELECT DISTINCT video_filepath FROM faces WHERE person_name IS NOT NULL
    ''').fetchall()

    for video in videos_to_tag:
        video_path = video['video_filepath']
        # Get all unique names associated with this video
        names = conn.execute('''
            SELECT DISTINCT person_name FROM faces WHERE video_filepath = ? AND person_name IS NOT NULL
        ''', (video_path,)).fetchall()

        person_names = [name['person_name'] for name in names]
        tags_string = ", ".join(person_names)

        print(f"Processing {video_path} -> Tags: {tags_string}")

        try:
            # Use ffmpeg to write metadata without re-encoding
            # This creates a temporary output file and then replaces the original
            input_path = video_path
            output_path = input_path + ".tagged.mp4"

            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(stream, output_path, c='copy', metadata=f'comment=People: {tags_string}')
            # The overwrite_output() is important if you run this multiple times
            ffmpeg.run(stream, overwrite_output=True, quiet=True)

            # Replace original file with the newly tagged one
            os.remove(input_path)
            os.rename(output_path, input_path)
            print(f"  - Successfully tagged and replaced file.")

        except Exception as e:
            print(f"  - ERROR tagging file {video_path}: {e}")

    conn.close()
    print("Metadata writing complete.")
    return redirect(url_for('index'))


if __name__ == '__main__':
    # Make sure the web app is accessible from other devices on your network
    app.run(host='0.0.0.0', port=5001, debug=True)
