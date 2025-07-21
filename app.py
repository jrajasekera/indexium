import os
import sqlite3
import sys

import ffmpeg
from flask import Flask, render_template_string, request, redirect, url_for, flash, send_file

# --- CONFIGURATION ---
# Get video directory from environment variable
VIDEO_DIRECTORY = os.environ.get("INDEXIUM_VIDEO_DIR")
if VIDEO_DIRECTORY is None:
    print("Error: INDEXIUM_VIDEO_DIR environment variable not set")
    print("Please set this variable to the directory containing your videos")
    sys.exit(1)

DATABASE_FILE = "video_faces.db"
# Directory containing cached face thumbnails
THUMBNAIL_DIR = "thumbnails"

app = Flask(__name__)
# Flask needs a secret key to use flash messages
app.secret_key = os.urandom(24)

# --- HTML TEMPLATES ---

# Base template with navigation and progress stats
BASE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Video Face Tagger{% endblock %}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style> 
        body { font-family: 'Inter', sans-serif; } 
        .face-checkbox:checked + img {
            border: 4px solid #3b82f6; /* blue-500 */
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
        }
    </style>
</head>
<body class="bg-gray-100 text-gray-800">
    <header class="bg-white shadow-sm sticky top-0 z-10">
        <div class="container mx-auto p-4 flex justify-between items-center">
            <nav class="flex items-center gap-6">
                <a href="{{ url_for('index') }}" class="text-lg font-bold text-gray-900 hover:text-blue-600">Home</a>
                <a href="{{ url_for('list_people') }}" class="text-lg font-bold text-gray-900 hover:text-blue-600">Review People</a>
            </nav>
            <div class="text-center">
                <span class="font-semibold">{{ stats.named_people_count }}</span> People Identified | 
                <span class="font-semibold">{{ stats.unnamed_groups_count }}</span> Groups Remaining
            </div>
            <form action="/write_metadata" method="post" onsubmit="return confirm('This will read and write metadata to your video files. This is a non-destructive process that creates new files, but please ensure you have backups. Are you sure?');">
                <button type="submit" class="bg-green-600 text-white font-semibold py-2 px-4 rounded-lg shadow-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-75 transition-colors">
                    Write All Named Tags
                </button>
            </form>
        </div>
    </header>
    <main class="container mx-auto p-4 sm:p-6 lg:p-8">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                <div class="mb-4 p-4 rounded-md {% if category == 'success' %}bg-green-100 text-green-800{% else %}bg-red-100 text-red-800{% endif %}">
                    {{ message }}
                </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        {% block content %}{% endblock %}
    </main>
</body>
</html>
'''

GROUP_TAGGER_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Tag Group #{{ cluster.id }}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style> 
        body { font-family: 'Inter', sans-serif; } 
        .face-checkbox:checked + img {
            border: 4px solid #3b82f6; /* blue-500 */
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
        }
    </style>
</head>
<body class="bg-gray-100 text-gray-800">
    <header class="bg-white shadow-sm sticky top-0 z-10">
        <div class="container mx-auto p-4 flex justify-between items-center">
            <nav class="flex items-center gap-6">
                <a href="{{ url_for('index') }}" class="text-lg font-bold text-gray-900 hover:text-blue-600">Home</a>
                <a href="{{ url_for('list_people') }}" class="text-lg font-bold text-gray-900 hover:text-blue-600">Review People</a>
            </nav>
            <div class="text-center">
                <span class="font-semibold">{{ stats.named_people_count }}</span> People Identified | 
                <span class="font-semibold">{{ stats.unnamed_groups_count }}</span> Groups Remaining
            </div>
            <form action="/write_metadata" method="post" onsubmit="return confirm('This will read and write metadata to your video files. This is a non-destructive process that creates new files, but please ensure you have backups. Are you sure?');">
                <button type="submit" class="bg-green-600 text-white font-semibold py-2 px-4 rounded-lg shadow-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-75 transition-colors">
                    Write All Named Tags
                </button>
            </form>
        </div>
    </header>
    <main class="container mx-auto p-4 sm:p-6 lg:p-8">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                <div class="mb-4 p-4 rounded-md {% if category == 'success' %}bg-green-100 text-green-800{% else %}bg-red-100 text-red-800{% endif %}">
                    {{ message }}
                </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        <div class="max-w-4xl mx-auto bg-white rounded-lg shadow-md p-6">
            <h2 class="font-semibold text-2xl mb-4 text-center">Who is this? (Group #{{ cluster.id }})</h2>

            <form id="face-selection-form" action="{{ url_for('split_cluster') }}" method="post">
                <input type="hidden" name="cluster_id" value="{{ cluster.id }}">
                <div class="flex flex-wrap justify-center gap-3 mb-6 p-4 bg-gray-50 rounded-lg">
                    {% for face in cluster.faces %}
                        <label class="cursor-pointer">
                            <input type="checkbox" name="face_ids" value="{{ face.id }}" class="hidden face-checkbox">
                            <img src="{{ url_for('get_face_thumbnail', face_id=face.id) }}" alt="Face from video" class="w-24 h-24 object-cover rounded-lg bg-gray-200 shadow transition-all">
                        </label>
                    {% else %}
                        <p>No faces found for this group. It may have been an error.</p>
                    {% endfor %}
                </div>
                <div class="text-center mb-6">
                    <button type="submit" class="bg-yellow-500 text-white font-semibold py-2 px-4 rounded-md hover:bg-yellow-600">Split Selected Faces into New Group</button>
                </div>
            </form>

            <!-- Naming Form -->
            <form action="{{ url_for('name_cluster') }}" method="post" class="mb-4 border-t pt-4">
                <input type="hidden" name="cluster_id" value="{{ cluster.id }}">
                <label for="person_name" class="block text-sm font-medium text-gray-700 mb-1">Name this entire group:</label>
                <div class="flex gap-2">
                    <input list="existing-names" id="person_name" name="person_name" placeholder="Enter or select a name..." class="flex-grow p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-blue-500" required autocomplete="off">
                    <datalist id="existing-names">
                        {% for name in existing_names %}
                            <option value="{{ name }}">
                        {% endfor %}
                    </datalist>
                    <button type="submit" class="bg-blue-600 text-white font-semibold py-2 px-4 rounded-md hover:bg-blue-700">Save Name</button>
                </div>
            </form>

            <!-- Merge Form -->
            <form action="{{ url_for('merge_clusters') }}" method="post" class="mb-4">
                <input type="hidden" name="from_cluster_id" value="{{ cluster.id }}">
                <label for="merge_target" class="block text-sm font-medium text-gray-700 mb-1">Or merge this group with an existing person:</label>
                <div class="flex gap-2">
                    <select name="to_person_name" id="merge_target" class="flex-grow p-2 border border-gray-300 rounded-md">
                        <option disabled selected>Select person to merge with...</option>
                        {% for name in existing_names %}
                            <option value="{{ name }}">{{ name }}</option>
                        {% endfor %}
                    </select>
                    <button type="submit" class="bg-purple-600 text-white font-semibold py-2 px-4 rounded-md hover:bg-purple-700">Merge</button>
                </div>
            </form>

            <!-- Actions -->
            <div class="flex justify-between items-center mt-6 border-t pt-4">
                <form action="{{ url_for('delete_cluster') }}" method="post" onsubmit="return confirm('Are you sure you want to permanently delete this group? This cannot be undone.');">
                    <input type="hidden" name="cluster_id" value="{{ cluster.id }}">
                    <button type="submit" class="text-sm text-red-600 hover:text-red-800">Delete Group</button>
                </form>
                <a href="{{ url_for('skip_cluster', cluster_id=cluster.id) }}" class="bg-gray-200 text-gray-800 font-semibold py-2 px-4 rounded-md hover:bg-gray-300">Skip →</a>
            </div>
        </div>
    </main>
</body>
</html>
'''

ALL_DONE_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>All Done!</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style> 
        body { font-family: 'Inter', sans-serif; } 
        .face-checkbox:checked + img {
            border: 4px solid #3b82f6; /* blue-500 */
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
        }
    </style>
</head>
<body class="bg-gray-100 text-gray-800">
    <header class="bg-white shadow-sm sticky top-0 z-10">
        <div class="container mx-auto p-4 flex justify-between items-center">
            <nav class="flex items-center gap-6">
                <a href="{{ url_for('index') }}" class="text-lg font-bold text-gray-900 hover:text-blue-600">Home</a>
                <a href="{{ url_for('list_people') }}" class="text-lg font-bold text-gray-900 hover:text-blue-600">Review People</a>
            </nav>
            <div class="text-center">
                <span class="font-semibold">{{ stats.named_people_count }}</span> People Identified | 
                <span class="font-semibold">{{ stats.unnamed_groups_count }}</span> Groups Remaining
            </div>
            <form action="/write_metadata" method="post" onsubmit="return confirm('This will read and write metadata to your video files. This is a non-destructive process that creates new files, but please ensure you have backups. Are you sure?');">
                <button type="submit" class="bg-green-600 text-white font-semibold py-2 px-4 rounded-lg shadow-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-75 transition-colors">
                    Write All Named Tags
                </button>
            </form>
        </div>
    </header>
    <main class="container mx-auto p-4 sm:p-6 lg:p-8">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                <div class="mb-4 p-4 rounded-md {% if category == 'success' %}bg-green-100 text-green-800{% else %}bg-red-100 text-red-800{% endif %}">
                    {{ message }}
                </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        <div class="max-w-2xl mx-auto bg-white p-8 rounded-lg shadow-md text-center">
            <h2 class="text-3xl font-semibold text-green-600">All Done!</h2>
            <p class="mt-4 text-gray-600">No more unnamed face groups found. You can run the scanner again, review the people you've already tagged, or write the tags to your video files.</p>
            <div class="mt-8 flex flex-col gap-4">
                <a href="{{ url_for('list_people') }}" class="w-full bg-blue-600 text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:bg-blue-700">Review Named People</a>
                <form action="/write_metadata" method="post" onsubmit="return confirm('This will write metadata to your video files. This is a non-destructive process that creates new files, but please ensure you have backups. Are you sure?');">
                    <button type="submit" class="w-full bg-green-600 text-white font-semibold py-3 px-4 rounded-lg shadow-md hover:bg-green-700">
                        Write All Named Tags to Files
                    </button>
                </form>
            </div>
        </div>
    </main>
</body>
</html>
'''

PEOPLE_LIST_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>All Identified People</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style> 
        body { font-family: 'Inter', sans-serif; } 
        .face-checkbox:checked + img {
            border: 4px solid #3b82f6; /* blue-500 */
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
        }
    </style>
</head>
<body class="bg-gray-100 text-gray-800">
    <header class="bg-white shadow-sm sticky top-0 z-10">
        <div class="container mx-auto p-4 flex justify-between items-center">
            <nav class="flex items-center gap-6">
                <a href="{{ url_for('index') }}" class="text-lg font-bold text-gray-900 hover:text-blue-600">Home</a>
                <a href="{{ url_for('list_people') }}" class="text-lg font-bold text-gray-900 hover:text-blue-600">Review People</a>
            </nav>
            <div class="text-center">
                <span class="font-semibold">{{ stats.named_people_count }}</span> People Identified | 
                <span class="font-semibold">{{ stats.unnamed_groups_count }}</span> Groups Remaining
            </div>
            <form action="/write_metadata" method="post" onsubmit="return confirm('This will read and write metadata to your video files. This is a non-destructive process that creates new files, but please ensure you have backups. Are you sure?');">
                <button type="submit" class="bg-green-600 text-white font-semibold py-2 px-4 rounded-lg shadow-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-75 transition-colors">
                    Write All Named Tags
                </button>
            </form>
        </div>
    </header>
    <main class="container mx-auto p-4 sm:p-6 lg:p-8">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                <div class="mb-4 p-4 rounded-md {% if category == 'success' %}bg-green-100 text-green-800{% else %}bg-red-100 text-red-800{% endif %}">
                    {{ message }}
                </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        <div class="max-w-4xl mx-auto bg-white rounded-lg shadow-md p-6">
            <h2 class="font-semibold text-2xl mb-6 text-center">Identified People ({{ people|length }})</h2>
            <div class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                {% for person in people %}
                <a href="{{ url_for('person_details', person_name=person.person_name) }}" class="block text-center p-2 bg-gray-50 rounded-lg hover:bg-blue-100 hover:shadow-lg transition-all">
                    <img src="{{ url_for('get_face_thumbnail', face_id=person.face_id) }}" alt="Face of {{ person.person_name }}" class="w-24 h-24 object-cover rounded-full mx-auto mb-2 shadow-md">
                    <p class="font-semibold text-gray-800 truncate">{{ person.person_name }}</p>
                    <p class="text-sm text-gray-500">{{ person.face_count }} faces</p>
                </a>
                {% else %}
                <p class="col-span-full text-center text-gray-500">No people have been named yet.</p>
                {% endfor %}
            </div>
        </div>
    </main>
</body>
</html>
'''

PERSON_DETAIL_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Details for {{ person_name }}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style> 
        body { font-family: 'Inter', sans-serif; } 
        .face-checkbox:checked + img {
            border: 4px solid #3b82f6; /* blue-500 */
            box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
        }
    </style>
</head>
<body class="bg-gray-100 text-gray-800">
    <header class="bg-white shadow-sm sticky top-0 z-10">
        <div class="container mx-auto p-4 flex justify-between items-center">
            <nav class="flex items-center gap-6">
                <a href="{{ url_for('index') }}" class="text-lg font-bold text-gray-900 hover:text-blue-600">Home</a>
                <a href="{{ url_for('list_people') }}" class="text-lg font-bold text-gray-900 hover:text-blue-600">Review People</a>
            </nav>
            <div class="text-center">
                <span class="font-semibold">{{ stats.named_people_count }}</span> People Identified | 
                <span class="font-semibold">{{ stats.unnamed_groups_count }}</span> Groups Remaining
            </div>
            <form action="/write_metadata" method="post" onsubmit="return confirm('This will read and write metadata to your video files. This is a non-destructive process that creates new files, but please ensure you have backups. Are you sure?');">
                <button type="submit" class="bg-green-600 text-white font-semibold py-2 px-4 rounded-lg shadow-md hover:bg-green-700 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-opacity-75 transition-colors">
                    Write All Named Tags
                </button>
            </form>
        </div>
    </header>
    <main class="container mx-auto p-4 sm:p-6 lg:p-8">
        {% with messages = get_flashed_messages(with_categories=true) %}
            {% if messages %}
                {% for category, message in messages %}
                <div class="mb-4 p-4 rounded-md {% if category == 'success' %}bg-green-100 text-green-800{% else %}bg-red-100 text-red-800{% endif %}">
                    {{ message }}
                </div>
                {% endfor %}
            {% endif %}
        {% endwith %}
        <div class="max-w-5xl mx-auto bg-white rounded-lg shadow-md p-6">
            <h2 class="font-semibold text-3xl mb-2 text-center">{{ person_name }}</h2>
            <p class="text-center text-gray-500 mb-6">{{ faces|length }} faces found for this person.</p>

            <div class="flex flex-wrap justify-center gap-3 mb-6 p-4 bg-gray-50 rounded-lg max-h-96 overflow-y-auto">
                {% for face in faces %}
                    <img src="{{ url_for('get_face_thumbnail', face_id=face.id) }}" alt="Face of {{ person_name }}" class="w-24 h-24 object-cover rounded-lg bg-gray-200 shadow">
                {% endfor %}
            </div>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 border-t pt-6">
                <!-- Rename Form -->
                <form action="{{ url_for('rename_person', old_name=person_name) }}" method="post">
                    <h3 class="font-semibold text-lg mb-2">Rename Person</h3>
                    <div class="flex gap-2">
                        <input type="text" name="new_name" placeholder="Enter new name..." class="flex-grow p-2 border border-gray-300 rounded-md" required>
                        <button type="submit" class="bg-blue-600 text-white font-semibold py-2 px-4 rounded-md hover:bg-blue-700">Rename</button>
                    </div>
                </form>

                <!-- Other Actions -->
                <div>
                    <h3 class="font-semibold text-lg mb-2">Other Actions</h3>
                    <div class="flex gap-4">
                        <form action="{{ url_for('unname_person') }}" method="post" onsubmit="return confirm('Are you sure you want to un-name this group? It will be sent back to the tagging queue.');">
                            <input type="hidden" name="person_name" value="{{ person_name }}">
                            <button type="submit" class="bg-yellow-500 text-white font-semibold py-2 px-4 rounded-md hover:bg-yellow-600">Un-name Group</button>
                        </form>
                        <form action="{{ url_for('delete_cluster_by_name') }}" method="post" onsubmit="return confirm('Are you sure you want to permanently delete this person and all their face data?');">
                            <input type="hidden" name="person_name" value="{{ person_name }}">
                            <button type="submit" class="bg-red-600 text-white font-semibold py-2 px-4 rounded-md hover:bg-red-700">Delete Person</button>
                        </form>
                    </div>
                </div>
            </div>
        </div>
    </main>
</body>
</html>
'''


# --- DATABASE & HELPERS ---

def get_db_connection():
    """Creates a database connection."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def get_progress_stats():
    """Gets counts for UI display."""
    conn = get_db_connection()
    unnamed_groups_count = conn.execute('''
        SELECT COUNT(DISTINCT cluster_id) as count 
        FROM faces 
        WHERE cluster_id IS NOT NULL AND person_name IS NULL
    ''').fetchone()['count']
    named_people_count = conn.execute('''
        SELECT COUNT(DISTINCT person_name) as count 
        FROM faces 
        WHERE person_name IS NOT NULL
    ''').fetchone()['count']
    conn.close()
    return {
        "unnamed_groups_count": unnamed_groups_count,
        "named_people_count": named_people_count
    }


# Make stats available to all templates
@app.context_processor
def inject_stats():
    return dict(stats=get_progress_stats())


# --- ROUTES ---

@app.route('/')
def index():
    """Finds the next unnamed group and redirects to the tagging page for it."""
    conn = get_db_connection()
    next_group = conn.execute('''
        SELECT MIN(cluster_id) as id
        FROM faces
        WHERE cluster_id IS NOT NULL AND person_name IS NULL
    ''').fetchone()
    conn.close()

    if next_group and next_group['id'] is not None:
        return redirect(url_for('tag_group', cluster_id=next_group['id']))
    else:
        # No more groups to name, show a completion page
        return render_template_string(ALL_DONE_TEMPLATE, BASE_TEMPLATE=BASE_TEMPLATE)


@app.route('/group/<int:cluster_id>')
def tag_group(cluster_id):
    """Displays a single group for tagging."""
    conn = get_db_connection()
    sample_faces = conn.execute('SELECT id FROM faces WHERE cluster_id = ? LIMIT 50', (cluster_id,)).fetchall()

    if not sample_faces:
        flash(f"Cluster #{cluster_id} no longer exists or is empty.", "error")
        return redirect(url_for('index'))

    names = conn.execute(
        'SELECT DISTINCT person_name FROM faces WHERE person_name IS NOT NULL ORDER BY person_name').fetchall()
    conn.close()

    existing_names = [name['person_name'] for name in names]
    cluster_data = {'id': cluster_id, 'faces': sample_faces}
    return render_template_string(GROUP_TAGGER_TEMPLATE, BASE_TEMPLATE=BASE_TEMPLATE, cluster=cluster_data,
                                  existing_names=existing_names)


@app.route('/face_thumbnail/<int:face_id>')
def get_face_thumbnail(face_id):
    """Serves a pre-generated face thumbnail."""
    thumb_path = os.path.join(THUMBNAIL_DIR, f"{face_id}.jpg")
    if os.path.exists(thumb_path):
        return send_file(thumb_path, mimetype='image/jpeg')
    return "Thumbnail not found", 404


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
        flash(f"Assigned name '{person_name}' to cluster #{cluster_id}", "success")
    return redirect(url_for('index'))


@app.route('/delete_cluster', methods=['POST'])
def delete_cluster():
    """Deletes all data associated with a cluster_id."""
    cluster_id = request.form['cluster_id']
    if cluster_id:
        conn = get_db_connection()
        conn.execute('DELETE FROM faces WHERE cluster_id = ?', (cluster_id,))
        conn.commit()
        conn.close()
        flash(f"Deleted all faces for cluster #{cluster_id}", "success")
    return redirect(url_for('index'))


@app.route('/skip_cluster/<int:cluster_id>')
def skip_cluster(cluster_id):
    """Finds the next available cluster with an ID greater than the current one."""
    conn = get_db_connection()
    next_group = conn.execute('''
        SELECT MIN(cluster_id) as id
        FROM faces
        WHERE cluster_id > ? AND cluster_id IS NOT NULL AND person_name IS NULL
    ''', (cluster_id,)).fetchone()
    conn.close()

    if next_group and next_group['id'] is not None:
        return redirect(url_for('tag_group', cluster_id=next_group['id']))
    else:
        flash("No more groups to skip to. Looping back to the start.", "success")
        return redirect(url_for('index'))


@app.route('/write_metadata', methods=['POST'])
def write_metadata():
    """
    Intelligently reads, merges, and writes face tags into the file's
    metadata 'comment' tag using ffmpeg.
    """
    print("Starting intelligent metadata write process...")
    conn = get_db_connection()
    videos_to_tag = conn.execute('SELECT DISTINCT file_hash FROM faces WHERE person_name IS NOT NULL').fetchall()

    tagged_count = 0
    for video in videos_to_tag:
        file_hash = video['file_hash']
        path_info = conn.execute('SELECT last_known_filepath FROM scanned_files WHERE file_hash = ?',
                                 (file_hash,)).fetchone()
        if not path_info or not os.path.exists(path_info['last_known_filepath']):
            print(f"  - WARNING: Path for hash {file_hash} not found. Skipping.")
            continue

        video_path = path_info['last_known_filepath']

        # Get names from DB
        db_names_rows = conn.execute(
            'SELECT DISTINCT person_name FROM faces WHERE file_hash = ? AND person_name IS NOT NULL',
            (file_hash,)).fetchall()
        db_names = {row['person_name'] for row in db_names_rows}

        # Get existing names from file metadata
        try:
            probe = ffmpeg.probe(video_path)
            comment_tag = probe.get('format', {}).get('tags', {}).get('comment', '')
            if 'People: ' in comment_tag:
                existing_names_str = comment_tag.split('People: ')[1]
                existing_names = {name.strip() for name in existing_names_str.split(',')}
            else:
                existing_names = set()
        except ffmpeg.Error as e:
            print(f"  - FFPROBE ERROR for {video_path}: {e.stderr.decode('utf8')}. Assuming no existing tags.")
            existing_names = set()

        # Merge names and create new tag string
        all_names = sorted(list(db_names.union(existing_names)))
        tags_string = ", ".join(all_names)

        print(f"Processing {video_path} -> Tags: {tags_string}")

        try:
            input_path = video_path
            output_path = os.path.join(os.path.dirname(input_path), f".temp_{os.path.basename(input_path)}")

            stream = ffmpeg.input(input_path)
            stream = ffmpeg.output(stream, output_path, c='copy', metadata=f'comment=People: {tags_string}')
            ffmpeg.run(stream, overwrite_output=True, quiet=True)

            os.remove(input_path)
            os.rename(output_path, input_path)
            print(f"  - Successfully tagged and replaced file.")
            tagged_count += 1
        except Exception as e:
            print(f"  - FFMPEG WRITE ERROR for file {video_path}: {e}")
            if os.path.exists(output_path): os.remove(output_path)

    conn.close()
    flash(f"Metadata writing complete. Updated {tagged_count} files.", "success")
    return redirect(url_for('index'))


# --- NEW ROUTES FOR REVIEWING/EDITING ---

@app.route('/people')
def list_people():
    """Shows a grid of all identified people."""
    conn = get_db_connection()
    # Get one representative face for each person, plus a count of their faces
    people = conn.execute('''
        SELECT p.person_name, p.face_count, f.id as face_id
        FROM (
            SELECT person_name, COUNT(id) as face_count, MIN(id) as min_face_id
            FROM faces
            WHERE person_name IS NOT NULL
            GROUP BY person_name
        ) p
        JOIN faces f ON p.min_face_id = f.id
        ORDER BY p.person_name
    ''').fetchall()
    conn.close()
    return render_template_string(PEOPLE_LIST_TEMPLATE, BASE_TEMPLATE=BASE_TEMPLATE, people=people)


@app.route('/person/<person_name>')
def person_details(person_name):
    """Shows all faces for one person and provides editing tools."""
    conn = get_db_connection()
    faces = conn.execute('SELECT id FROM faces WHERE person_name = ?', (person_name,)).fetchall()
    conn.close()
    if not faces:
        flash(f"Person '{person_name}' not found.", "error")
        return redirect(url_for('list_people'))
    return render_template_string(PERSON_DETAIL_TEMPLATE, BASE_TEMPLATE=BASE_TEMPLATE, person_name=person_name,
                                  faces=faces)


@app.route('/rename_person/<old_name>', methods=['POST'])
def rename_person(old_name):
    new_name = request.form['new_name'].strip()
    if new_name:
        conn = get_db_connection()
        conn.execute('UPDATE faces SET person_name = ? WHERE person_name = ?', (new_name, old_name))
        conn.commit()
        conn.close()
        flash(f"Renamed '{old_name}' to '{new_name}'.", "success")
        return redirect(url_for('person_details', person_name=new_name))
    else:
        flash("New name cannot be empty.", "error")
        return redirect(url_for('person_details', person_name=old_name))


@app.route('/unname_person', methods=['POST'])
def unname_person():
    person_name = request.form['person_name']
    conn = get_db_connection()
    conn.execute('UPDATE faces SET person_name = NULL WHERE person_name = ?', (person_name,))
    conn.commit()
    conn.close()
    flash(f"'{person_name}' has been un-named and their group is back in the queue.", "success")
    return redirect(url_for('list_people'))


@app.route('/delete_cluster_by_name', methods=['POST'])
def delete_cluster_by_name():
    person_name = request.form['person_name']
    conn = get_db_connection()
    # This is safer than deleting by cluster_id, as a person might be a result of merges
    conn.execute('DELETE FROM faces WHERE person_name = ?', (person_name,))
    conn.commit()
    conn.close()
    flash(f"Deleted person '{person_name}' and all their face data.", "success")
    return redirect(url_for('list_people'))


# --- NEW ROUTES FOR MERGE/SPLIT ---

@app.route('/merge_clusters', methods=['POST'])
def merge_clusters():
    from_cluster_id = request.form['from_cluster_id']
    to_person_name = request.form.get('to_person_name')

    if not to_person_name:
        flash("You must select a person to merge with.", "error")
        return redirect(url_for('tag_group', cluster_id=from_cluster_id))

    conn = get_db_connection()
    # Find the cluster_id of the person we are merging into
    target_cluster = conn.execute('SELECT cluster_id FROM faces WHERE person_name = ? LIMIT 1',
                                  (to_person_name,)).fetchone()
    if not target_cluster:
        conn.close()
        flash(f"Could not find target person '{to_person_name}'.", "error")
        return redirect(url_for('tag_group', cluster_id=from_cluster_id))

    to_cluster_id = target_cluster['cluster_id']

    # Reassign the old cluster_id and set the name
    conn.execute('UPDATE faces SET cluster_id = ?, person_name = ? WHERE cluster_id = ?',
                 (to_cluster_id, to_person_name, from_cluster_id))
    conn.commit()
    conn.close()

    flash(f"Successfully merged group #{from_cluster_id} into '{to_person_name}'.", "success")
    return redirect(url_for('index'))


@app.route('/split_cluster', methods=['POST'])
def split_cluster():
    original_cluster_id = request.form['cluster_id']
    face_ids_to_split = request.form.getlist('face_ids')

    if not face_ids_to_split:
        flash("You must select at least one face to split into a new group.", "error")
        return redirect(url_for('tag_group', cluster_id=original_cluster_id))

    conn = get_db_connection()
    # Find the highest existing cluster_id to create a new one
    max_cluster_id_result = conn.execute("SELECT MAX(cluster_id) FROM faces").fetchone()
    new_cluster_id = (max_cluster_id_result[0] or 0) + 1

    # Create a placeholder string for the SQL query
    placeholders = ', '.join('?' for _ in face_ids_to_split)
    query = f"UPDATE faces SET cluster_id = ? WHERE id IN ({placeholders})"

    params = [new_cluster_id] + face_ids_to_split
    conn.execute(query, params)
    conn.commit()
    conn.close()

    flash(
        f"Successfully split {len(face_ids_to_split)} faces from group #{original_cluster_id} into new group #{new_cluster_id}.",
        "success")
    return redirect(url_for('tag_group', cluster_id=original_cluster_id))


if __name__ == '__main__':
    # Make sure the web app is accessible from other devices on your network
    app.run(host='0.0.0.0', port=5001, debug=True)
