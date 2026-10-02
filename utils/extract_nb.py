import json

notebooks = [
    'model/brain_tumor_clahe_shap.ipynb',
    'notebook.ipynb'
]

for nb_path in notebooks:
    try:
        with open(nb_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        print('=== NOTEBOOK:', nb_path, '===')
        print('Total cells:', len(nb['cells']))
        print()
        
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'markdown':
                src = ''.join(cell['source'])
                if src.strip():
                    print('--- MARKDOWN Cell', i, '---')
                    print(src[:600])
                    print()
            elif cell['cell_type'] == 'code':
                outputs = cell.get('outputs', [])
                for out in outputs:
                    otype = out.get('output_type', '')
                    if otype == 'stream':
                        text = ''.join(out.get('text', []))
                        if text.strip():
                            print('--- CODE Cell', i, 'OUTPUT ---')
                            print(text[:1500])
                            print()
                    elif otype == 'execute_result':
                        data = out.get('data', {})
                        text = ''.join(data.get('text/plain', []))
                        if text.strip() and len(text) < 2000:
                            print('--- CODE Cell', i, 'RESULT ---')
                            print(text[:1500])
                            print()
    except Exception as e:
        print('Error reading', nb_path, ':', e)
    print()
    print('=' * 60)
    print()
