import json
with open('vit_finetune.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('vit_finetune_cells.py', 'w', encoding='utf-8') as f:
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            f.write(f"# --- Cell {i} ---\n")
            f.write(''.join(cell['source']))
            f.write('\n\n')
