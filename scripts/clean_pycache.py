"""Delete all __pycache__ directories and .pyc files in the project."""
import os
import shutil

root    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
removed = 0

for dirpath, dirnames, filenames in os.walk(root):
    parts = dirpath.replace("\\", "/").split("/")
    if ".venv" in parts or ".git" in parts:
        dirnames.clear()
        continue

    if "__pycache__" in dirnames:
        target = os.path.join(dirpath, "__pycache__")
        shutil.rmtree(target)
        print(f"  removed {os.path.relpath(target, root)}")
        dirnames.remove("__pycache__")
        removed += 1

    for f in filenames:
        if f.endswith(".pyc"):
            path = os.path.join(dirpath, f)
            os.remove(path)
            print(f"  removed {os.path.relpath(path, root)}")
            removed += 1

print(f"\ndone — {removed} item(s) removed")
