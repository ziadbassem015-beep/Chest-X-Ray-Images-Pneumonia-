Chest X-Ray Pneumonia Detection using CNN (ResNet50)
🩺 Project Overview

This project aims to detect pneumonia from chest X-ray images using a Convolutional Neural Network (CNN) with transfer learning. The model uses a ResNet50 backbone pretrained on ImageNet and a custom classifier on top for binary classification (NORMAL vs PNEUMONIA).

🗂 Dataset

Source: Chest X-Ray Images (Pneumonia) dataset

Structure:

chest_xray/
    train/
        NORMAL/
        PNEUMONIA/
    test/
        NORMAL/
        PNEUMONIA/


Total images:

Training: 4173

Validation: 1043

Testing: 624

⚙️ Dependencies

Python 3.x

TensorFlow / Keras

NumPy

Pandas

Matplotlib

Seaborn

scikit-learn

You can install dependencies via:

pip install tensorflow numpy pandas matplotlib seaborn scikit-learn

🖼️ Data Preprocessing

Images rescaled to [0,1].

Augmentation applied to training set:

Rotation, zoom, width/height shifts, horizontal flip

Split: 80% training, 20% validation

🧠 Model Architecture

Base model: ResNet50 (ImageNet pretrained, frozen)

Custom top layers:

Global Average Pooling

Batch Normalization

Dropout (0.4)

Dense (256, ReLU)

Dropout (0.2)

Output layer: Dense(1, Sigmoid)

Loss function: Binary Crossentropy

Metrics: Accuracy, Precision, Recall

Class weights applied to handle class imbalance

🚀 Training

Epochs: 10

Optimizer: Adam

Callbacks:

EarlyStopping (monitor val_loss)

ReduceLROnPlateau

ModelCheckpoint

Example training snippet:

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=10,
    class_weight=class_weight_dict,
    callbacks=callbacks
)

📊 Performance

Test set results:

              precision    recall  f1-score   support
NORMAL       0.89      0.79      0.84       234
PNEUMONIA    0.88      0.94      0.91       390
accuracy                           0.88       624


Confusion Matrix visualized with Seaborn heatmap

Accuracy: 88%

💾 Model

Saved as: medical_pneumonia_model_final1.keras

Load the model for inference:

from tensorflow.keras.models import load_model
model = load_model('medical_pneumonia_model_final1.keras')

🔗 Usage

Clone the repo:

git clone <repo_url>


Install dependencies

Run training or inference scripts

Predict pneumonia on new chest X-ray images

📈 Visualization

Sample images per class

Class distribution bar plot

Confusion matrix heatmap

⚠️ Notes

Make sure your dataset folder structure matches the expected layout.

This project uses transfer learning, so training time is faster than training from scratch.

Adjust batch size or learning rate for different GPU configurations.
