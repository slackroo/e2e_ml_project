import logging
import os
from datetime import datetime


def find_project_root():
    """
    Searches for a project root marked by the presence of a 'README.md' file,
    starting from the current working directory.

    :return: The path to the root directory of the project.
    """
    # Start from the current working directory
    current_dir = os.getcwd()

    # Look for a distinctive file or directory to mark the project root
    root_marker = 'README.md'  # Change this to your project's distinctive marker

    # Traverse up until we find the root marker or hit the file system root
    while True:
        files_and_dirs = os.listdir(current_dir)
        parent_dir = os.path.dirname(current_dir)
        if root_marker in files_and_dirs:
            # Found the project root
            return current_dir
        elif parent_dir == current_dir:
            # Hit the file system root without finding the marker
            raise FileNotFoundError("Project root could not be found.")
        else:
            # Move up one level
            current_dir = parent_dir


# Usage
project_root = find_project_root()

LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
logs_path = os.path.join(project_root, "logs")
os.makedirs(logs_path, exist_ok=True)

LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,

)

# if __name__ == "__main__":
#     logging.info("Logging has started")
