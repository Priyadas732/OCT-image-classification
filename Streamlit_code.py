import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# Load the trained model
MODEL_PATH = "D:/neurothon/saved_model/model.pth"
NUM_CLASSES = 4  # Update this if needed
dic  = {0:"CNV",1:"DME",2:"DRUSEN",3:"NORMAL"}
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found! Please check the path.")
        return None
    
    try:
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, NUM_CLASSES)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        model.eval()
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.title("OCR Image Classification")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    # Load model and make prediction
    model = load_model()
    if model is not None:
        try:
            with torch.no_grad():
                output = model(img_tensor)
                predicted_class = torch.argmax(output, dim=1).item()
            
            st.write(f"Predicted Class: {dic[predicted_class]}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
