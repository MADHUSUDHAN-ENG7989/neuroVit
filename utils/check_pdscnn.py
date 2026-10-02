import json

with open('pdscnn_base_paper.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = "".join(cell.get('source', []))
    if cell['cell_type'] == 'markdown':
        print(f"--- Cell {i} (Markdown) ---")
        print(src[:100])
    elif "DRIVE_PROCESSED_ROOT =" in src:
        print(f"--- Cell {i} (Code - PROCESSED_ROOT) ---")
    elif "def compute_mean_std" in src:
        print(f"--- Cell {i} (Code - compute_mean_std) ---")
