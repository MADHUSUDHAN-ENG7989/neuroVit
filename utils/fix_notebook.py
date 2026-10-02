import json
import os

nb_path = 'pdscnn_base_paper.ipynb'
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Modify cells
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        # 1. Image size and speedups
        if 'IMG_SIZE =' in source:
            # Change to 224 for higher accuracy, or keep 124 but maybe that's why accuracy is low.
            # The user complained about accuracy being too low (needs >= 99%) and running slow.
            # Let's use IMG_SIZE = 224 but increase batch size and use prefetch to speed up.
            source = source.replace('IMG_SIZE = 124', 'IMG_SIZE = 224')
            source = source.replace('BATCH_SIZE = 32', 'BATCH_SIZE = 64') # Speed up
            cell['source'] = [line + '\n' for line in source.split('\n')]
            cell['source'][-1] = cell['source'][-1].strip('\n')
            
        # 2. Saving preprocessed images to drive
        if 'PROCESSED_ROOT = \'/content/processed\'' in source:
            save_code = """
import shutil
DRIVE_PROCESSED_ROOT = '/content/drive/MyDrive/processed_data_vit'
print("Copying preprocessed images to Drive for ViT training...")
shutil.copytree(PROCESSED_ROOT, DRIVE_PROCESSED_ROOT, dirs_exist_ok=True)
print("Saved preprocessed images to Drive!")
"""
            if 'DRIVE_PROCESSED_ROOT' not in source:
                source = source + "\n" + save_code
            cell['source'] = [line + '\n' for line in source.split('\n')]
            cell['source'][-1] = cell['source'][-1].strip('\n')

        # 3. Model Architecture update for better accuracy
        if 'class PDSCNN(nn.Module):' in source:
            # Add more capacity or skip connections if possible, but simplest is keeping widths larger
            # and dropping dropout rate slightly or adding more epochs if needed, but user said they run 100 epochs.
            # Let's replace the PDSCNN with a deeper or wider version if it's the custom one.
            if 'self.branch_11 =' in source:
                # It's the custom patched one
                new_model = source.replace('self.dropout = nn.Dropout(0.5)', 'self.dropout = nn.Dropout(0.3)')
                new_model = new_model.replace('256 * 5, 128', '256 * 5, 256')
                new_model = new_model.replace('self.bn6 = nn.BatchNorm2d(128)', 'self.bn6 = nn.BatchNorm2d(256)')
                new_model = new_model.replace('128, 64', '256, 128')
                new_model = new_model.replace('self.bn7 = nn.BatchNorm2d(64)', 'self.bn7 = nn.BatchNorm2d(128)')
                new_model = new_model.replace('64, 32', '128, 64')
                new_model = new_model.replace('self.bn8 = nn.BatchNorm2d(32)', 'self.bn8 = nn.BatchNorm2d(64)')
                new_model = new_model.replace('32, 16', '64, 32')
                new_model = new_model.replace('self.bn9 = nn.BatchNorm2d(16)', 'self.bn9 = nn.BatchNorm2d(32)')
                # FC input size will change! We need to handle that or use AdaptiveAvgPool2d
                pool_code = """        x = self.pool9(self.bn9(self.relu(self.conv9(x))))
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = self.flatten(x)"""
                new_model = new_model.replace("""        x = self.pool9(self.bn9(self.relu(self.conv9(x))))
        
        x = self.flatten(x)""", pool_code)
                new_model = new_model.replace('self.fc1 = nn.Linear(400, 512)', 'self.fc1 = nn.Linear(32, 512)')
                cell['source'] = [line + '\n' for line in new_model.split('\n')]
                cell['source'][-1] = cell['source'][-1].strip('\n')

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)

print("Notebook modified successfully.")
