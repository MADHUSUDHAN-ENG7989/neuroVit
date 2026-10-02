import json

def patch_notebook():
    with open('vit_finetune.ipynb', 'r', encoding='utf-8') as f:
        nb = json.load(f)

    new_cells = []
    
    for i, cell in enumerate(nb['cells']):
        src = "".join(cell.get('source', []))
        
        if "DRIVE_PROCESSED_ROOT = '/content/drive/MyDrive/processed_data_vit'" in src:
            new_code = """from google.colab import drive
import os, glob
import pandas as pd
import shutil

drive.mount('/content/drive')

DRIVE_PROCESSED_ROOT = '/content/drive/MyDrive/processed_data_vit'
LOCAL_PROCESSED_ROOT = '/content/processed_data_vit'

# Copy the processed data to the fast local disk of the Colab instance to prevent I/O bottlenecks during training
if not os.path.exists(LOCAL_PROCESSED_ROOT):
    print("Copying data from Google Drive to local disk for faster training... This may take a minute or two.")
    shutil.copytree(DRIVE_PROCESSED_ROOT, LOCAL_PROCESSED_ROOT)
    print("Copy complete!")

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}

records = []
for cls in CLASS_NAMES:
    cls_dir = os.path.join(LOCAL_PROCESSED_ROOT, cls)
    if os.path.exists(cls_dir):
        for fname in os.listdir(cls_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                records.append({'processed_path': os.path.join(cls_dir, fname), 'label': cls})

df = pd.DataFrame(records)
print(f"Loaded {len(df)} preprocessed images directly from {LOCAL_PROCESSED_ROOT}")
print(df['label'].value_counts())
"""
            new_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [line + '\n' for line in new_code.split('\n')]
            })
        else:
            new_cells.append(cell)
            
    nb['cells'] = new_cells
    
    with open('vit_finetune.ipynb', 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

if __name__ == '__main__':
    patch_notebook()
    print("Notebook patched.")
