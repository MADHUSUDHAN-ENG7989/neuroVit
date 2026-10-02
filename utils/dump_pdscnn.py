import json

with open('pdscnn_base_paper.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Total cells: {len(nb['cells'])}")
print()

for i, cell in enumerate(nb['cells']):
    src = "".join(cell.get('source', []))
    if cell['cell_type'] == 'code':
        print(f"=== Cell {i} (CODE) ===")
        print(src[:800])
        print()
