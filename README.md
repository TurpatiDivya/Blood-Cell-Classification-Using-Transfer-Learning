
## 🧪 **Blood Cell Classification Using Transfer Learning with VGG19**

### 🔷 **Abstract**

Accurate classification of white blood cells plays a vital role in early diagnosis and monitoring of diseases such as leukemia, anemia, and infections. In this project, we present a deep learning-based approach for automatic classification of white blood cells using **transfer learning** with the **VGG19** architecture. The model is trained to classify four major types of blood cells — **EOSINOPHIL, LYMPHOCYTE, MONOCYTE**, and **NEUTROPHIL** — with high accuracy and efficiency.

---

### 🔷 **Introduction**

Manual classification of white blood cells under a microscope is time-consuming, error-prone, and requires expert knowledge. Automation using deep learning offers a scalable and consistent solution. Given the limited size of medical image datasets, we adopt a **transfer learning approach**, leveraging the pretrained **VGG19** model originally trained on ImageNet to extract high-level image features and retrain it for domain-specific classification.

---

### 🔷 **Methodology**

#### ✅ **Dataset**

The dataset comprises labeled images of white blood cells categorized into four classes:

* **EOSINOPHIL**
* **LYMPHOCYTE**
* **MONOCYTE**
* **NEUTROPHIL**

The data is split into training, validation, and test sets using a folder-based structure.

#### ✅ **Model Architecture**

* **Base Model**: VGG19 (pretrained on ImageNet)
* **Modification**: The final classification layer is replaced with a custom linear layer suited for 4 classes.
* **Training Strategy**:

  * Early convolutional layers are frozen to retain pretrained features.
  * Final layers are fine-tuned to adapt to blood cell characteristics.
  * Additional regularization (dropout, weight decay) is used to avoid overfitting.

#### ✅ **Transformations**

Images are resized to $224 \times 224$, normalized, and augmented during training using:

* Random horizontal flips
* Random rotations
* Color jitter

#### ✅ **Evaluation**

* Model accuracy is evaluated on the test set.
* Confusion matrix is plotted to analyze class-wise performance.
* Visual predictions with confidence scores are generated using matplotlib.

---

### 🔷 **Results**

* **Training Accuracy**: > 99.43%
* **Test Accuracy**: \~80% (can be improved with more data/fine-tuning)
* **Prediction Visualization**: The model correctly predicts classes with reasonable confidence and robustness to minor variations.

---

### 🔷 **Conclusion**

This project demonstrates the effectiveness of transfer learning using VGG19 for medical image classification tasks with limited data. By leveraging pretrained feature extractors and fine-tuning on a domain-specific dataset, high accuracy can be achieved without extensive training from scratch. Future improvements include applying more advanced models (ResNet, EfficientNet), hyperparameter tuning, and addressing class imbalance.

---

### 🔷 **Keywords**

Transfer Learning, VGG19, White Blood Cell Classification, Deep Learning, EOSINOPHIL, LYMPHOCYTE, MONOCYTE, NEUTROPHIL, Medical Image Analysis.

