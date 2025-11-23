import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

DATA_DIR = "data"
IMAGE_SIZE = (64, 64)
NUM_CLASSES = len(os.listdir(DATA_DIR))

def train_cnn_model():
    X = []
    y = []
    classes = os.listdir(DATA_DIR)

    print("Loading images...")
    for idx, class_name in enumerate(classes):
        class_folder = os.path.join(DATA_DIR, class_name)
        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)
            image = Image.open(file_path).convert("RGB")
            image = image.resize(IMAGE_SIZE)
            X.append(np.array(image)/255.0) 
            y.append(idx)

    X = np.array(X)
    y = to_categorical(np.array(y), NUM_CLASSES)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(64,64,3)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    print("Training CNN model...")
    history = model.fit(X_train, y_train, epochs=10, batch_size=16, validation_data=(X_test, y_test))

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.2f}")

    os.makedirs("models", exist_ok=True)
    model.save("models/cnn_model.h5")
    print("CNN model saved in 'models/cnn_model.h5'")
    return model, acc, cm, X_test, y_test


if __name__ == "__main__":
    train_cnn_model()
