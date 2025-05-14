import os
from collections import defaultdict
import yaml

# Starting nav structure
nav = [{"Home": "index.md"}]

# Organize files by top-level folder after docs/
docs_dir = "docs"
folder_map = defaultdict(list)

# Traverse docs/imgraph/ and group files
for root, _, files in os.walk(os.path.join(docs_dir, "imgraph")):
    for file in sorted(files):
        if file.endswith(".md"):
            rel_dir = os.path.relpath(root, docs_dir)
            rel_path = os.path.join(rel_dir, file).replace("\\", "/")
            parts = rel_path.split("/")
            if len(parts) >= 3:  # e.g., imgraph/data/edge_creation.md
                section = parts[1].replace("_", " ").title()
                label = parts[-1].replace(".md", "").replace("_", " ").title()
                folder_map[section].append({label: rel_path})
            else:
                label = parts[-1].replace(".md", "").replace("_", " ").title()
                folder_map["Misc"].append({label: rel_path})

# Add sorted sections to nav
for section, pages in sorted(folder_map.items()):
    nav.append({section: pages})

# Output YAML nav structure
nav_yaml = yaml.dump({"nav": nav}, sort_keys=False, width=120)

# Save to mkdocs_nav.yml for preview or copy
with open("mkdocs_nav.yml", "w") as f:
    f.write(nav_yaml)

nav_yaml

