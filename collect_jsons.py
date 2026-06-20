import os
import shutil
import re

def collect_jsons():
    # 1. Define the directory structure
    source_dir = os.getcwd()
    dest_dir = os.path.join(source_dir, "all_jsons")
    
    # 2. Regex pattern provided
    pattern = r"([^_]+)_(\d+)_([^_]+(?:_[^_]+)*)_\(([^)]+)\)\.json$"

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    print(f"Searching in: {source_dir}")
    print(f"Collecting to: {dest_dir}\n")

    # 3. Walk through the directory tree
    for root, dirs, files in os.walk(source_dir):
        # Ignore the destination folder to prevent infinite loops
        if "all_jsons" in dirs:
            dirs.remove("all_jsons")
            
        for file in files:
            if file.endswith(".json"):
                # 4. Validate filename against regex
                if re.match(pattern, file):
                    source_file = os.path.join(root, file)
                    dest_file = os.path.join(dest_dir, file)
                    
                    shutil.copy2(source_file, dest_file)
                    print(f"MATCH: Collected '{file}'")
                else:
                    # Optional: Print files that didn't match the format
                    # print(f"SKIP: '{file}' does not match format")
                    pass

    print("\nCollection complete.")

if __name__ == "__main__":
    collect_jsons()