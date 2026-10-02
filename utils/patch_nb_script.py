import json

def patch_notebook():
    with open('vit_finetune.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = []
    
    for i, cell in enumerate(nb['cells']):
        if i == 6:
            # Replace cell 6 with new direct loading code
            new_code = """from google.colab import drive
import glob, os
import pandas as pd

drive.mount('/content/drive')

DRIVE_PROCESSED_ROOT = '/content/drive/MyDrive/processed_data_vit'
# OR it could be processed_data based on what they used.
# Let's collect them directly
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

records = []
for cls in CLASS_NAMES:
    cls_dir = os.path.join(DRIVE_PROCESSED_ROOT, cls)
    if os.path.exists(cls_dir):
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                records.append({'processed_path': os.path.join(cls_dir, fname), 'label': cls})

df = pd.DataFrame(records)
print(f"Loaded {len(df)} preprocessed images directly from {DRIVE_PROCESSED_ROOT}")
print(df['label'].value_counts())
"""
            new_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + '\n' for line in new_code.split('\n')]
            })
        elif 6 < i <= 25:
            # Skip all these preprocessing cells
            pass
        elif i == 8 or i == 9 or i == 10 or i == 11 or i == 12 or i == 13 or i == 14 or i == 15 or i == 16 or i == 17 or i == 18 or i == 19 or i == 20 or i == 21 or i == 22 or i == 23 or i == 24 or i == 25:
            # wait, the index might shift if I skip markdown too.
            # It's better to filter based on content or skip a range.
            pass
        else:
            pass # we'll implement a better logic

if __name__ == '__main__':
    patch_notebook()
