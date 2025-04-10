![PyTorch Logo](https://github.com/pytorch/pytorch/raw/main/docs/source/_static/img/pytorch-logo-dark.png)

# Image Classification Web App with ResNet & PyTorch

## Overview

This repository hosts a Streamlit-based web application for image classification using a Convolutional Neural Network (CNN) built on the ResNet architecture. The model, implemented in PyTorch, is designed to classify medical images into one of four categories: CNV, DME, DRUSEN, and NORMAL. The app provides an interactive user interface where users can upload images, and it displays the predicted class based on the processed input.

## Table of Contents

  - [Features](#features)
  - [Architecture and Technology Stack](#architecture-and-technology-stack)
  - [Installation](#installation)
      - [Prerequisites](#prerequisites)
      - [Required Python Packages](#required-python-packages)
      - [Clone the Repository](#clone-the-repository)
      - [Virtual Environment Setup (Optional but Recommended)](#virtual-environment-setup-optional-but-recommended)
      - [Install Dependencies](#install-dependencies)
  - [Setup](#setup)
      - [Model Preparation](#model-preparation)
          - [Model File Location](#model-file-location)
          - [Model Architecture](#model-architecture)
  - [Usage](#usage)
      - [Running the Application](#running-the-application)
  - [Code Walkthrough](#code-walkthrough)
      - [Streamlit UI and Image Upload](#streamlit-ui-and-image-upload)
      - [Image Preprocessing](#image-preprocessing)
      - [Model Loading and Prediction](#model-loading-and-prediction)
      - [Error Handling](#error-handling)
  - [Troubleshooting](#troubleshooting)
  - [Future Enhancements](#future-enhancements)
  - [Contributing](#contributing)
  - [License](#license)

## Features

  * **User-Friendly Interface:** Built with Streamlit, allowing users to effortlessly upload images and see results.
  * **Robust Image Processing:** Leverages standard PyTorch transformations to preprocess images, ensuring consistent input for the model.
  * **State-of-the-Art Model:** Utilizes a fine-tuned ResNet18 architecture for accurate image classification.
  * **Error Handling:** Includes checks for the existence of the model file and gracefully handles prediction errors.
  * **Customizable:** Easy to update the model file path, adjust the number of classes, or extend the image transformation pipeline.

## Architecture and Technology Stack

  * **Front-End:** [Streamlit](https://streamlit.io/) is used to build an interactive and responsive web UI.
  * **Back-End:** [PyTorch](https://pytorch.org/) powers the deep learning model, providing flexibility for customization and experimentation.
  * **Model Architecture:** The app employs a ResNet18 model, with a custom fully connected layer to output predictions for four target classes.
  * **Image Processing:** Uses [torchvision transforms](https://pytorch.org/vision/stable/transforms.html) for resizing, normalizing, and converting images into tensors suitable for model inference.
  * **Dependencies:** Python 3.6+, PyTorch, Torchvision, Streamlit, and Pillow.

## Installation

### Prerequisites

  * [Python](https://www.python.org/) 3.6 or later
  * [pip](https://pip.pypa.io/en/stable/) (Python package installer)

### Required Python Packages

  * [torch](https://pytorch.org/)
  * [torchvision](https://pytorch.org/vision/stable/)
  * [streamlit](https://streamlit.io/)
  * [pillow](https://pypi.org/project/Pillow/)

### Clone the Repository -->

Open your terminal and run:

```bash
git clone https://github.com/Abhay-2309/OCT-Image-Classification.git
cd your-repo-name



##License

This project is open source and available under the [MIT License](LICENSE). Feel free to use, modify, and distribute the code as per the terms of the license.

