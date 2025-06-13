import json
import os

def save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)
