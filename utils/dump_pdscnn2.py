import json

with open('pdscnn_base_paper.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Print cells 2, 3, 5, 7 in full
for i in [2, 3, 5, 7]:
    cell = nb['cells'][i]
    src = "".join(cell.get('source', []))
    print(f"=== Cell {i} ({cell['cell_type'].upper()}) ===")
    print(src)
    print()
