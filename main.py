import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import cv2
import zipfile
import urllib.request

# Download EuroSAT dataset
def download_dataset():
    if not os.path.exists("data/2750"):
        print("Please download and extract EuroSAT manually into 'data/2750'")
        exit("Dataset not found.")

# Load and preprocess images
def load_data(img_size=(64, 64)):
    data_dir = "data/2750"
    classes = os.listdir(data_dir)
    X, y = [], []
    label_map = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        cls_folder = os.path.join(data_dir, cls)
        for img_name in os.listdir(cls_folder)[:500]:
            img_path = os.path.join(cls_folder, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(label_map[cls])

    X = np.array(X, dtype='float32') / 255.0
    y = to_categorical(np.array(y))
    return X, y, classes

# Build a CNN model
def create_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Main
if __name__ == "__main__":
    download_dataset()
    X, y, classes = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_model(X.shape[1:], len(classes))
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.2f}")

    # Visualize some predictions
    preds = model.predict(X_test[:5])
    for i, pred in enumerate(preds):
        plt.imshow(X_test[i])
        plt.title(f"Predicted: {classes[np.argmax(pred)]}")
        plt.axis("off")
        plt.show()
