import json

with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell.get('source', []))
    print(f"Cell {i} ({cell['cell_type']}): {source[:50].replace(chr(10), ' ')}")
