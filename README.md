# Fruit and Vegetable Quality Detection

## Overview
Fruit and Vegetable Quality Detection is a **Machine Learning / Computer Vision project** that classifies the quality of fruits and vegetables using image data. The system analyzes images and determines whether the produce is **fresh or rotten**, helping automate quality inspection in agriculture and food supply chains.

This project demonstrates how computer vision can improve efficiency in food quality monitoring and agricultural processing.

---

## Features
- Detects fruits and vegetables from uploaded images
- Classifies produce quality (**Fresh / Rotten**)
- Uses **Machine Learning / Deep Learning models**
- Image preprocessing for improved prediction accuracy
- Simple and easy-to-use prediction pipeline

---

## Technologies Used

### Programming Language
- Python

### Libraries
- TensorFlow / Keras
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn

### Tools
- Jupyter Notebook
- Git
- GitHub

---

## Dataset

The dataset contains images of various fruits and vegetables used to train the model to classify quality.

The dataset is divided into:

- **Training Set** – used to train the model  
- **Validation Set** – used for tuning and improving model performance  
- **Test Set** – used to evaluate the final model performance  

---

## Machine Learning Pipeline

1. **Data Collection**
   - Collect images of fruits and vegetables.

2. **Data Preprocessing**
   - Image resizing
   - Normalization
   - Data augmentation

3. **Model Training**
   - Train a **Convolutional Neural Network (CNN)** to classify fruit and vegetable quality.

4. **Model Evaluation**
   - Evaluate the model using metrics such as **accuracy and loss**.

5. **Prediction**
   - Use the trained model to predict the quality of fruits and vegetables from new images.

---

## Installation

### Clone the repository

```bash
git clone https://github.com/AmanGupta1354/Fruit-and-Vegetable-Quality-Detection-.git
````

### Navigate to the project directory

```bash
cd Fruit-and-Vegetable-Quality-Detection-
```

### Install required dependencies

```bash
pip install -r requirements.txt
```

---

## Usage

Run the prediction script:

```bash
python predict.py
```

Upload an image of a fruit or vegetable and the model will classify its quality.

Example output:

```
Prediction: Fresh Apple
Confidence: 94%
```

---

## Applications

* Automated agricultural quality inspection
* Smart farming solutions
* Food processing industries
* Retail quality monitoring
* Supply chain automation

---

## Future Improvements

* Deploy as a **web application using Flask or Streamlit**
* Add **more fruit and vegetable classes**
* Improve model accuracy using **Transfer Learning**
* Enable **real-time detection using webcam**
* Deploy as a **mobile application**

---

