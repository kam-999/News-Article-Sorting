import os
from pathlib import Path

list_of_files = [
    f"BBC_News/__init__.py",
    f"BBC_News/cloud_storage/__init__.py",
    f"BBC_News/components/__init__.py",
    f"BBC_News/configuration/__init__.py",
    f"BBC_News/constants/__init__.py",
    f"BBC_News/data_access/__init__.py",
    f"BBC_News/entity/__init__.py",
    f"BBC_News/exceptions/__init__.py",
    f"BBC_News/logger/__init__.py",
    f"BBC_News/pipeline/__init__.py",
    f"BBC_News/utils/__init__.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
    else:
        print(f"file is already present at: {filepath}")