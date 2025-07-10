import os
import cv2
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load images and labels
data = []
labels = []

def load_images_from_folder(folder, label):
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            resized = cv2.resize(img, (48, 48))     # Resize if needed
            flattened = resized.flatten()           # 48x48 → 2304 pixel values
            data.append(flattened)
            labels.append(label)

# Load Happy images (label = 1)
load_images_from_folder("dataset/Happy", 1)

# Load Sad images (label = 0)
load_images_from_folder("dataset/Sad", 0)

# Convert to numpy arrays
X = np.array(data)
y = np.array(labels)

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save model as .pkl
os.makedirs("model", exist_ok=True)
with open("model/emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model/emotion_model.pkl")
