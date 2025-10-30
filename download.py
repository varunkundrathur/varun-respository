import kagglehub
import shutil
import os

# Desired save path
save_path = r"D:\VARUN\programming files\vs code\python\new fashion item generator\Dataset"

# Download dataset using KaggleHub
download_path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")

# Move dataset to your custom folder
if not os.path.exists(save_path):
    os.makedirs(save_path)

for item in os.listdir(download_path):
    s = os.path.join(download_path, item)
    d = os.path.join(save_path, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)

print("Dataset downloaded and saved to:", save_path)
