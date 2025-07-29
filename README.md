# Diabetic Retinopathy Detection Using Deep Learning

This repository contains a deep learning-based system for detecting Diabetic Retinopathy (DR) from retinal fundus images. The model classifies images into five severity levels and is optimized for high accuracy, interpretability, and real-world deployment.

## Features

- Multi-class classification using ResNet-50, EfficientNetB4/B5, and ensemble learning
- Achieved 96.44% accuracy, 98.45% precision, and 95.70% F1-score on the APTOS 2019 dataset
- Integrated explainable AI techniques (Grad-CAM, SHAP) to improve model transparency
- Optimized for real-time inference using GPU and lightweight deployment on edge devices
- Trained using transfer learning, data augmentation, and multi-task loss strategies

## Dataset

- **Source**: APTOS 2019 Blindness Detection (Kaggle)
- **Classes**:
  - 0: No DR
  - 1: Mild
  - 2: Moderate
  - 3: Severe
  - 4: Proliferative DR

## Model Architecture

- Base model: ResNet-50 (pre-trained on ImageNet)
- Ensemble models: EfficientNetB4 and EfficientNetB5
- Loss functions: Cross-Entropy, Focal Loss, Mean Squared Error
- Optimization: SGD and Rectified Adam with Cosine Annealing LR Scheduler
- Regularization: Label smoothing and noise injection
- Explainability: Grad-CAM and SHAP visualizations

## Technology Stack

- Python
- PyTorch, Catalyst
- OpenCV, NumPy, Scikit-learn, Matplotlib
- SHAP for model explainability
- Jupyter Notebook / Google Colab for experimentation

## Results

| Metric     | Value     |
|------------|-----------|
| Accuracy   | 96.44%    |
| Precision  | 98.45%    |
| Recall     | 93.50%    |
| F1 Score   | 95.70%    |

## Folder Structure

├── data/ # Contains raw and processed images
├── models/ # Trained model checkpoints
├── notebooks/ # Jupyter notebooks for training and evaluation
│ ├── 1_data_preprocessing.ipynb
│ ├── 2_model_training.ipynb
│ └── 3_explainability.ipynb
├── src/ # Core scripts and utilities
│ ├── model.py
│ ├── train.py
│ ├── eval.py
│ └── utils.py
├── README.md

# Clone the project

bash
Copy
Edit
git clone https://github.com/yourusername/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection
Install required libraries

# Make sure Python 3.7+ is installed, then run:

bash
Copy
Edit
pip install -r requirements.txt
# Download the dataset

Go to: APTOS 2019 Blindness Detection – Kaggle

Download the dataset files (train_images, test_images, train.csv, test.csv)

Place them in the data/ folder like this:

kotlin
Copy
Edit
data/
├── train_images/
├── test_images/
├── train.csv
└── test.csv
# Train the model

You can use either the script or the notebook:

Using script:

bash
Copy
Edit
python src/train.py
Using Jupyter Notebook:

Open and run all cells in:

Copy
Edit
notebooks/2_model_training.ipynb
Evaluate the model

Using script:

bash
Copy
Edit
python src/eval.py
Or open the evaluation notebook:

Copy
Edit
notebooks/3_explainability.ipynb
Run prediction on a new image

Use the command below, replacing the path with your image file:

bash
Copy
Edit
python src/infer.py --image path/to/your_image.png
