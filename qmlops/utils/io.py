# src/utils/io.py
import os
import json

def ensure_dirs(*dirs):
    """Make sure each directory in dirs exists."""
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def save_json(path, obj):
    """Save a Python object as JSON."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
