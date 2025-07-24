import os
import sqlite3
import math

import ffmpeg
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    send_file,
    g,
    session,
)

from config import Config

config = Config()

app = Flask(__name__)
# Flask needs a secret key to use flash messages
app.secret_key = config.SECRET_KEY

# --- DATABASE & HELPERS ---

def get_db_connection():
    """Gets a per-request database connection."""
    if "db" not in g:
        g.db = sqlite3.connect(config.DATABASE_FILE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db_connection(exception):
    """Closes the database connection at the end of the request."""
    db = g.pop("db", None)
    if db is not None:
        db.close()


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
    return {
        "unnamed_groups_count": unnamed_groups_count,
        "named_people_count": named_people_count
    }


# Make stats available to all templates
@app.context_processor
def inject_stats():
    return dict(stats=get_progress_stats())


# Routes are now organized in blueprints
from blueprints.tagging_bp import tagging_bp
from blueprints.people_bp import people_bp

app.register_blueprint(tagging_bp)
app.register_blueprint(people_bp)
if __name__ == '__main__':
    # Make sure the web app is accessible from other devices on your network
    app.run(host='0.0.0.0', port=5001, debug=config.DEBUG)
