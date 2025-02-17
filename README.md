Image Classification Web App with ResNet & PyTorch
Overview
This repository hosts a Streamlit-based web application for image classification using a Convolutional Neural Network (CNN) built on the ResNet architecture. The model, implemented in PyTorch, is designed to classify medical images into one of four categories: CNV, DME, DRUSEN, and NORMAL. The app provides an interactive user interface where users can upload images, and it displays the predicted class based on the processed input.

Table of Contents
Features
Architecture and Technology Stack
Installation
Prerequisites
Required Python Packages
Clone the Repository
Virtual Environment Setup (Optional but Recommended)
Install Dependencies
Setup
Model Preparation
Model File Location
Model Architecture
Usage
Running the Application
Code Walkthrough
Streamlit UI and Image Upload
Image Preprocessing
Model Loading and Prediction
Error Handling
Troubleshooting
Future Enhancements
Contributing
License
Features
User-Friendly Interface: Built with Streamlit, allowing users to effortlessly upload images and see results.
Robust Image Processing: Leverages standard PyTorch transformations to preprocess images, ensuring consistent input for the model.
State-of-the-Art Model: Utilizes a fine-tuned ResNet18 architecture for accurate image classification.
Error Handling: Includes checks for the existence of the model file and gracefully handles prediction errors.
Customizable: Easy to update the model file path, adjust the number of classes, or extend the image transformation pipeline.
Architecture and Technology Stack
Front-End: Streamlit is used to build an interactive and responsive web UI.
Back-End: PyTorch powers the deep learning model, providing flexibility for customization and experimentation.
Model Architecture: The app employs a ResNet18 model, with a custom fully connected layer to output predictions for four target classes.
Image Processing: Uses torchvision transforms for resizing, normalizing, and converting images into tensors suitable for model inference.
Dependencies: Python 3.6+, PyTorch, Torchvision, Streamlit, and Pillow.
Installation
Prerequisites
Python 3.6 or later
pip (Python package installer)
Required Python Packages
torch
torchvision
streamlit
pillow
Clone the Repository
Open your terminal and run:

Bash

git clone [https://github.com/Abhay-2309/OCT-Image-Classification.git](https://github.com/Abhay-2309/OCT-Image-Classification.git)
cd your-repo-name
Virtual Environment Setup (Optional but Recommended)
Create and activate a virtual environment to isolate project dependencies:

Bash

python -m venv venv
source venv/bin/activate  # On Linux/macOS
venv\Scripts\activate  # On Windows
Install Dependencies
You can install the required packages using pip:

Bash

pip install torch torchvision streamlit pillow
Alternatively, if you have a requirements.txt file, install dependencies with:

Bash

pip install -r requirements.txt
Setup
Model Preparation
Model File Location
The application expects the trained model to be located at:

Bash

D:/neurothon/saved_model/model.pth
If your model is saved elsewhere, update the MODEL_PATH variable in the Streamlit_code.py file accordingly.

Model Architecture
The loaded model is a modified ResNet18. The final fully connected layer has been replaced to match the number of target classes (default: 4). This adjustment ensures that the model outputs are compatible with the specific classification task.

Usage
Running the Application
Start the Streamlit application by running the following command in your terminal:

Bash

streamlit run Streamlit_code.py
This command will launch the app and open a new browser window or tab displaying the user interface. From there, you can:

Upload an Image: Use the file uploader widget to select and upload an image (accepted formats: JPG, JPEG, PNG).
View Prediction: Once the image is uploaded, the app processes the image, feeds it through the ResNet model, and displays the predicted class along with the uploaded image.
Code Walkthrough
Streamlit UI and Image Upload
UI Initialization: The app starts with a title and a file uploader widget that accepts images.
Image Display: When an image is uploaded, it is immediately displayed on the page with a caption.
Image Preprocessing
Transformations: The image is resized to 224x224 pixels, converted to a tensor, and normalized using standard mean and standard deviation values. This preprocessing ensures that the input image meets the requirements of the ResNet model.
Model Loading and Prediction
load_model Function: This function checks for the existence of the model file. It then loads a pre-trained ResNet18 model, replaces its final fully connected layer, and loads the saved weights.
Inference: The processed image is passed through the model within a torch.no_grad() context to prevent gradient computations. The model's output is used to determine the class with the highest score, which is then mapped to a human-readable label.
Error Handling
Model File Check: The application displays an error message if the model file is not found.
Prediction Error: Any exceptions during prediction are caught and reported, ensuring the app does not crash unexpectedly.
Troubleshooting
Model File Not Found:
Ensure that the model file exists at the specified path. If the file is missing, download or move the file to the correct location.
Dependency Issues:
Verify that all required Python packages are installed. Use the provided requirements.txt file to install any missing dependencies.
Unexpected Errors:
If you encounter any errors during prediction or UI interaction, check the Streamlit logs for detailed error messages.
Future Enhancements
Dynamic Model Loading: Allow users to upload their own trained models via the UI.
Extended Preprocessing Options: Add more customizable preprocessing options for different image sizes and formats.
Real-Time Updates: Implement features to update or retrain the model directly from the web interface.
Performance Optimization: Explore optimizations for model inference to reduce latency, especially for larger models.
Contributing
Contributions are welcome! If you'd like to contribute, please follow these steps:

Fork the Repository: Click the fork button at the top right of the repository page.

Create a Branch: Create a new branch for your feature or bug fix:

Bash

git checkout -b feature-name
Commit Your Changes: Commit your changes with clear and descriptive commit messages.

Submit a Pull Request: Open a pull request detailing your changes and the issue it addresses.

For major changes, please open an issue first to discuss what you would like to change.

License
This project is open source and available under the MIT License. Feel free to use, modify, and distribute the code as per the terms of the license.