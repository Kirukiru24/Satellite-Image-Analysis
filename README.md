
# 🛰️ Satellite Image Analysis using Deep Learning

This project demonstrates a simple yet effective deep learning pipeline for classifying satellite images using the [EuroSAT dataset](https://github.com/phelber/eurosat). The goal is to recognize different land use and land cover types from satellite imagery using a Convolutional Neural Network (CNN) implemented in TensorFlow.

## 📌 Project Overview

Satellite imagery provides critical data for a variety of domains, including agriculture, urban planning, environmental monitoring, and disaster management. This project focuses on:

- Loading and preprocessing the EuroSAT RGB dataset.
- Building a CNN model using TensorFlow/Keras.
- Training and evaluating the model on satellite image classes.
- Visualizing classification performance.

---

## 🧠 Model Architecture

The model is a basic CNN with:

- Convolutional layers
- MaxPooling
- Dropout for regularization
- Fully-connected layers for classification

Trained on 64x64 RGB satellite images across 10 classes.

---

## 📂 Dataset

- **Name**: EuroSAT RGB
- **Source**: [EuroSAT GitHub](https://github.com/phelber/eurosat)
- **Classes**: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

---

## 🛠️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Kirukiru24/Satellite-Image-Analysis.git
cd Satellite-Image-Analysis
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** TensorFlow 2.x is required. Use Python 3.9 for compatibility.

### 4. Run the Script

```bash
python main.py
```

---

## 📊 Results

- Achieved high classification accuracy on the test set.
- Sample visualizations show predictions on unseen satellite images.

> Detailed accuracy and loss graphs are generated after training.

---

## 🚫 .gitignore

Make sure the following are excluded from version control:

```gitignore
venv/
__pycache__/
*.pyc
*.zip
*.log
```

---

## ✅ Requirements

- Python 3.9
- TensorFlow 2.x
- NumPy
- OpenCV
- scikit-learn
- Matplotlib

---

## 📘 License

This project is licensed under the MIT License. See `LICENSE` for details.

---

## 🙌 Acknowledgments

- [EuroSAT Dataset](https://github.com/phelber/eurosat)
- [TensorFlow](https://www.tensorflow.org/)
- [scikit-learn](https://scikit-learn.org/)

---

## 📬 Contact

For questions or suggestions, feel free to open an issue or reach out to the project maintainer:

**GitHub**: [@Kirukiru24](https://github.com/Kirukiru24)

⬇
