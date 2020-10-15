import os
        
def list_files(path):
    files = os.listdir(path)
    return [file for file in files if not file.startswith('.')]

def is_empty(path):
    files = list_files(path)
    return len(files) > 0
    
def delete_files_not_folders(path):
    """Delete all files in directory and subdirectories."""
    for root, dirs, files in os.walk(path):
        for file in files:
            if not file.startswith('.'):
                os.remove(os.path.join(root, file))