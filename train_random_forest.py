import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

DATA_DIR = "data"
IMAGE_SIZE = (64, 64) 

def train_rf_model():
  
    X = []
    y = []
    classes = os.listdir(DATA_DIR) 

    for idx, class_name in enumerate(classes):
        class_folder = os.path.join(DATA_DIR, class_name)
        for file_name in os.listdir(class_folder):
            file_path = os.path.join(class_folder, file_name)
            image = Image.open(file_path).convert("RGB")
            image = image.resize(IMAGE_SIZE)
            X.append(np.array(image).flatten()) 
            y.append(idx)

    X = np.array(X)
    y = np.array(y)

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    
    y_pred = rf_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc}")
    print("Confusion Matrix:")
    print(cm)

    
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, "models/rf_model.pkl")
    print("Random Forest model saved in 'models/rf_model.pkl'")

    return acc, cm, X_test, y_test



if __name__ == "__main__":
    train_rf_model()
