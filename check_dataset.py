import os

DATASET_DIR = r"C:\tomato_doctor\dataset-tomatoes"

if not os.path.exists(DATASET_DIR):
    print(f"[ERROR] Directory '{DATASET_DIR}' not found!")
else:
    print(f"Checking dataset directory: {DATASET_DIR}\n")
    folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
    print(f"Found {len(folders)} class folders:")
    total_images = 0
    for folder in folders:
        folder_path = os.path.join(DATASET_DIR, folder)
        images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        count = len(images)
        total_images += count
        print(f"  - {folder}: {count} images")
    print(f"\nTotal images found: {total_images}")