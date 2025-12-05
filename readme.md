# 🩺 Skin Lesions Detection Using Deep Learning

This project focuses on building a deep learning model capable of classifying pigmented skin lesions from dermatoscopic images. The goal is to assist early diagnosis of melanoma and other skin abnormalities using automated image analysis.

---

## 📁 Project Structure
```
Skin-Lesions-Detection-
│                # Images used for training & testing
├── notebooks/               # Jupyter notebooks for EDA, training & evaluation                 # Saved model weights
                   # Python source code (preprocessing, training, prediction)
├── results/                 # Accuracy, loss curves, confusion matrix, predictions
└── README.md                # Project documentation
```

---

## 📘 Overview
Skin cancer is one of the most common and dangerous cancers if not diagnosed early.  
Using convolutional neural networks (CNNs), this project classifies dermatoscopic skin images into different categories of lesions.

The project includes:
- Exploratory Data Analysis  
- Image preprocessing & augmentation  
- Training CNN-based classifiers  
- Fine-tuning pre-trained models  
- Evaluating performance  
- Visualizing predictions  

---

## 🧠 Models Implemented
- **Baseline CNN model**
- **VGG16 (fine-tuned)**
- **InceptionV3 (fine-tuned)**
- **DenseNet201 (fine-tuned)**
- **Inception-ResNet V2 (fine-tuned)**
- **Ensemble models**

---

## 📊 Results Summary

| Model | Validation Accuracy | Test Accuracy | Depth | Parameters |
|-------|----------------------|----------------|--------|------------|
| Baseline CNN | 77.48% | 76.54% | 11 layers | 2.1M |
| VGG16 (fine-tuned) | 79.82% | 79.64% | 23 layers | 14.9M |
| InceptionV3 (fine-tuned) | 79.93% | 79.94% | 315 layers | 22.8M |
| Inception-ResNet V2 (fine-tuned) | 80.82% | 82.53% | 784 layers | 55.1M |
| DenseNet201 (fine-tuned) | **85.8%** | **83.9%** | 711 layers | 19.3M |
| InceptionV3 (full fine-tuning) | 86.92% | 86.82% | — | — |
| DenseNet201 (full fine-tuning) | **86.69%** | **87.72%** | — | — |
| Ensemble (InceptionV3 + DenseNet201) | **88.8%** | **88.52%** | — | — |

---

## ⚙️ Technical Notes

In older Keras/TensorFlow versions, Batch Normalization layers may behave inconsistently during fine-tuning.  
A workaround is:

```python
for layer in pre_trained_model.layers:
    if hasattr(layer, 'moving_mean') and hasattr(layer, 'moving_variance'):
        layer.trainable = True
        K.eval(K.update(layer.moving_mean, K.zeros_like(layer.moving_mean)))
        K.eval(K.update(layer.moving_variance, K.zeros_like(layer.moving_variance)))
    else:
        layer.trainable = False
```

---

## 🧪 How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Train the model
```bash
python src/train.py
```

### 3️⃣ Evaluate the model
```bash
python src/evaluate.py
```

### 4️⃣ Predict on a new image
```bash
python src/predict.py --image path/to/image.jpg
```

---

## 📈 Visualizations  
You may include visual outputs such as:
- Training & validation accuracy curves  
- Confusion matrix  
- Sample predictions  
- Grad-CAM heatmaps  

(Add images in the `results/` folder and link them here.)

---

## 🙌 Contributions
Feel free to fork the repository, improve model training, or add new architectures.

