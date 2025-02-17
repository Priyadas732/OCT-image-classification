import torch
import torchvision.transforms.functional as F
from PIL import Image
import os

# Input and output directories
image_dir = r"D:\neurothon\Dataset_2\val\NORMAL"  
output_dir = r"D:\neurothon\Dataset_2\val\NORMAL"  

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Sharpness factor (>1 increases, <1 decreases)
sharpness_factor = 4.0  

# Process images
for image_name in os.listdir(image_dir):
    image_path = os.path.join(image_dir, image_name)

    # Open image
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image)  # Convert to tensor

    # Apply sharpness adjustment
    sharpened_tensor = F.adjust_sharpness(image_tensor, sharpness_factor)
    sharpened_image = F.to_pil_image(sharpened_tensor)  # Convert back

    # Save output
    sharpened_image.save(os.path.join(output_dir, image_name))

print("Processing complete. Sharpened images saved to:", output_dir)
