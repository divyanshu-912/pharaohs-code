<h1 align="center">Pharaohs Code</h1>
<p align="center">
</p>
<a href="https://weekendofcode.computercodingclub.in/"> <img src="https://i.postimg.cc/njCM24kx/woc.jpg" height=30px> </a>

## Introduction:
The Pharaoh CAPTCHA Solver is a machine learning-based system designed to recognize and decode text-based CAPTCHAs using TensorFlow (Keras), OpenCV, and Flask. It employs a Convolutional Neural Network (CNN) for character recognition. 
The system includes:
Image Preprocessing – Converts CAPTCHA images to grayscale, applies noise reduction, binarization, and segments individual characters.
CNN Model Training – A deep learning model is trained on CAPTCHA images, normalizing input and using categorical cross-entropy for multi-class classification.
CAPTCHA Generation – Uses the ImageCaptcha library to create synthetic training data.
Flask API for Decoding – Provides a REST API to upload CAPTCHA images and return the decoded text in real-time.
The project can be trained locally and deployed as a web service, making it suitable for automating CAPTCHA recognition in various applications. 
## Table of Contents:
1. Introduction
Overview of the Pharaoh CAPTCHA Solver

2. Project Structure
Explanation of the key files and directories

3. Image Preprocessing
Grayscale conversion, noise reduction, and binarization
Character segmentation

4. Model Training
CNN architecture for CAPTCHA recognition
Data loading and preprocessing
Training and saving the model

5. CAPTCHA Generation
Generating synthetic CAPTCHA images for training

6. Flask API for CAPTCHA Decoding
API endpoint for image upload and text extraction
Model loading and prediction
7. Deployment
Running the Flask application
Hosting on a cloud service
8. Usage & Testing
How to test the model and API
Example inputs and outputs

9. Future Improvements
Enhancements for better accuracy and outputs
Training model shall be trained 
 🚀
## Technology Stack:
  1) Visual Studio Code
  2) Python
  3) Kaggle
  4) Machine Learning
  

## Contributors:

Team Name: Punk 327

* [Divyanshu Kannaujiya](https://github.com/divyanshu-912)
* [Shubham Saini](https://github.com/cyberpunk2005)
* [Piyush Chundawat](#)


## Project Structure
pharaohs-code/
├── backend/
│ ├── models/
│ │ ├── init.py
│ │ └── captcha_model.py
│ ├── utils/
│ │ ├── init.py
│ │ ├── data_generator.py
│ │ └── image_processing.py
│ ├── init.py
│ ├── app.py
│ └── config.py
├── frontend/
│ ├──
│ │── pharaohs_bg2.jpg
│ ├── css/
│ │ └── style.css
│ ├── js/
│ │ └── main.js
│ └── index.html
├── data/
│ └── captcha_dataset/
│     ├── train/
│     ├── validation/
│     └── test/
└── train.py

### Made at:



<a href="[https://hack36.com](https://weekendofcode.computercodingclub.in/)"> <img src="https://i.postimg.cc/mrCCnTbN/tpg.jpg" height=30px> </a>
