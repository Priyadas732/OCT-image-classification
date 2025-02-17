import streamlit as st
import torch
import torchvision.transforms.functional as F
from PIL import Image

def process_image(image: Image.Image, sharpness_factor: float = 4.0) -> Image.Image:
    """
    Simulate a deep learning model processing the image.
    Here, we adjust the image's sharpness as a demonstration.
    
    Parameters:
    - image (PIL.Image.Image): Input image.
    - sharpness_factor (float): Factor to adjust sharpness. (>1 increases, <1 decreases)
    
    Returns:
    - PIL.Image.Image: Processed image.
    """
    # Convert the PIL image to a PyTorch tensor
    image_tensor = F.to_tensor(image)
    
    # Perform the transformation (this represents your model's inference step)
    processed_tensor = F.adjust_sharpness(image_tensor, sharpness_factor)
    
    # Convert the processed tensor back to a PIL image
    processed_image = F.to_pil_image(processed_tensor)
    return processed_image

def main():
    st.title("Deep Learning Model Deployment")
    st.write("Upload an image to process it using the model.")

    # File uploader for input image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the uploaded image
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, caption="Input Image", use_column_width=True)
        
        # Process the image using the model (or image processing function)
        output_image = process_image(input_image, sharpness_factor=4.0)
        
        st.image(output_image, caption="Output Image", use_column_width=True)

if __name__ == "__main__":
    main()
