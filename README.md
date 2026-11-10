# Real-Time Sign Language Interpreter 

A computer vision and deep learning-based system for recognising American Sign Language (ASL) alphabet gestures using hand landmark detection and image classification.

The project uses **PyTorch, MediaPipe and OpenCV** to recognise static hand gestures and perform real-time predictions through a webcam.

##  Project Overview

The system explores two approaches to ASL alphabet recognition:

1. **CNN-based Image Classification:** A Convolutional Neural Network trained to classify ASL alphabet images.
2. **Landmark-based Recognition:** A neural network that uses 63 hand landmark features extracted through MediaPipe.

The landmark-based model is integrated with a webcam interface for real-time gesture recognition.

##  Features

- ASL alphabet recognition (A–Z).
- Additional classes: Space and Delete.
- Hand detection and landmark extraction using MediaPipe.
- 63-dimensional hand landmark feature representation.
- Wrist-relative coordinate normalisation.
- Real-time webcam-based inference.
- Top-3 prediction confidence display.
- CNN and landmark-based model implementations.

##  Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | PyTorch |
| Computer Vision | OpenCV, MediaPipe |
| Data Processing | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Visualisation | Matplotlib, Seaborn |
| Development | Jupyter Notebook, VS Code |

##  Model Architecture

### 1. CNN-based Model

A Convolutional Neural Network trained on ASL alphabet images to learn visual features and classify hand gestures.

- Training dataset: 60,900 images
- Validation dataset: 13,050 images
- Test dataset: 13,050 images
- Best validation accuracy: 99.88%
- Held-out test accuracy: **99.84%**
- Number of classes: 29

### 2. Landmark-based Model

MediaPipe Hands extracts 21 hand landmarks, with each landmark containing X, Y and Z coordinates.

This produces 63 numerical features per hand.

**Preprocessing:**
- Extract 21 hand landmarks.
- Normalise coordinates relative to the wrist.
- Apply scale normalisation using the maximum landmark distance.
- Flatten the coordinates into a 63-dimensional feature vector.

**Neural Network Architecture:**

- Input layer: 63 features
- Hidden layer 1: 128 neurons
- Hidden layer 2: 64 neurons
- Output layer: 28 classes
- Activation: ReLU
- Regularisation: Dropout and Batch Normalisation

**Evaluation Results:**

| Metric | Result |
|---|---:|
| Test Accuracy | 99.03% |
| Macro F1-score | 0.9889 |
| Weighted F1-score | 0.9903 |

*Note: These results represent held-out dataset evaluation, not real-world webcam accuracy.*

The landmark model uses 28 classes (A–Z, Space and Delete). The absence of a detected hand is handled separately as "Nothing".

##  Project Structure

```text
sign-language-interpreter/
│
├── app/
├── data/
│   ├── processed/
│   ├── landmarks/
│   └── raw/
│
├── models/
│   ├── asl_cnn_best.pth
│   └── asl_landmark_best.pth
│
├── notebooks/
│   └── ASL_CNN_Training.ipynb
│
├── results/
│
├── screenshots/
│
├── src/
│   ├── data/
│   ├── features/
│   ├── inference/
│   ├── models/
│   └── utils/
│
├── requirements.txt
├── README.md
└── .gitignore
```

##  Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Uttra21/sign-language-interpreter.git
cd sign-language-interpreter
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/macOS:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the webcam interpreter

Make sure your trained landmark model is available at:

```text
models/asl_landmark_best.pth
```

Run:

```bash
python src/inference/landmark_webcam.py
```

Allow webcam access and show ASL hand gestures in front of the camera.

Press **Q** to exit.

##  Results

The project achieved:

- 99.84% held-out test accuracy using the CNN model.
- 99.03% held-out test accuracy using the landmark-based model.
- Real-time webcam inference using MediaPipe and PyTorch.

These are separate evaluations of the two models and should not be interpreted as webcam recognition accuracy.

##  Future Improvements

- Improve real-world webcam recognition consistency.
- Implement temporal prediction smoothing.
- Develop sentence formation from recognised gestures.
- Integrate text-to-speech functionality.
- Extend the system towards continuous sign language recognition.

##  Author

**Uttra Manhas**

B.Tech – Computer Science and Engineering

GitHub: [@Uttra21](https://github.com/Uttra21)
