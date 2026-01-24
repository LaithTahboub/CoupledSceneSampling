import os
import shutil
from pathlib import Path

source_dir = "/fs/nexus-scratch/ltahboub/CoupledSceneSampling/test_final/Mysore_Palace/images/commons"
output_dir = "/fs/nexus-scratch/ltahboub/CoupledSceneSampling/train_ready_data"
valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

count = 0

for root, dirs, files in os.walk(source_dir):
    if os.path.basename(root) == "pictures":
        for file in files:
            file_path = Path(file)
            if file_path.suffix.lower() in valid_extensions:
                src_path = os.path.join(root, file)
                new_name = f"mysore_palace_{count:04d}{file_path.suffix.lower()}"
                dst_path = os.path.join(output_dir, new_name)

                shutil.copy2(src_path, dst_path)
                count += 1

print(f"Successfully prepared {count} images in {output_dir}")
