import json

with open('pdscnn_base_paper.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i in [11, 14, 16, 17]:
    cell = nb['cells'][i]
    src = "".join(cell.get('source', []))
    print(f"=== Cell {i} ({cell['cell_type'].upper()}) ===")
    print(src)
    print()
