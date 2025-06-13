import os
import zipfile

def export_workspace(user_id):
    folder = f"workspace/"
    zip_path = f"exports/{user_id}_project.zip"
    os.makedirs("exports", exist_ok=True)
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        for root, dirs, files in os.walk(folder):
            for file in files:
                full_path = os.path.join(root, file)
                zipf.write(full_path, full_path)
    return zip_path
