import os
import shutil

# Define source and destination directories
source_dir = r"D:\neurothon\Dataset_2\train\NORMAL"
dest_dir = r"D:\neurothon\Dataset_2\train\NORMAL2"

# Create destination directory if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)

# Define image file extensions to filter
image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff")

# Get a list of image files from the source directory
all_files = os.listdir(source_dir)
image_files = [f for f in all_files if f.lower().endswith(image_extensions)]

# Select 2000 images (if there are at least 2000, otherwise select all)
selected_images = image_files[:2000]

# Copy each selected image to the destination directory
for image in selected_images:
    src_path = os.path.join(source_dir, image)
    dst_path = os.path.join(dest_dir, image)
    shutil.copy(src_path, dst_path)

print(f"Copied {len(selected_images)} images from {source_dir} to {dest_dir}.")
