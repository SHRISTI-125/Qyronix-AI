import os
import torch
import torch.nn.functional as F

from PIL import Image

from torchvision import transforms
from torchvision import models

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# DEVICE
# =========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =========================================
# IMAGE TRANSFORM
# =========================================

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# =========================================
# LOAD MODEL
# =========================================

model = models.resnet18(weights=None)

model.fc = torch.nn.Sequential(

    torch.nn.Dropout(0.5),

    torch.nn.Linear(
        model.fc.in_features,
        2
    )
)

# =========================================
# MODEL PATH
# =========================================

model_path = os.path.join(
    "models",
    "skin_model.pth"
)

# =========================================
# LOAD WEIGHTS
# =========================================

model.load_state_dict(

    torch.load(
        model_path,
        map_location=device
    )
)

model.to(device)

model.eval()

# =========================================
# DATASET PATH
# =========================================

dataset_path = "datasets/test"

# Folder Structure:
#
# test/
#    benign/
#    malignant/

# =========================================
# LABELS
# =========================================

class_names = [
    "benign",
    "malignant"
]

label_map = {

    "benign":0,

    "malignant":1
}

# =========================================
# STORE RESULTS
# =========================================

y_true = []

y_pred = []

# =========================================
# PREDICTION LOOP
# =========================================

with torch.no_grad():

    for class_name in class_names:

        folder_path = os.path.join(
            dataset_path,
            class_name
        )

        for file in os.listdir(folder_path):

            img_path = os.path.join(
                folder_path,
                file
            )

            try:

                img = Image.open(img_path).convert("RGB")

                x = transform(img).unsqueeze(0).to(device)

                out = model(x)

                prob = F.softmax(out, dim=1)

                _, pred = torch.max(prob,1)

                y_true.append(
                    label_map[class_name]
                )

                y_pred.append(
                    pred.item()
                )

            except Exception as e:

                print(f"Error in {file} --> {e}")

# =========================================
# ACCURACY
# =========================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print(f"\nAccuracy : {accuracy*100:.2f}%")

# =========================================
# CONFUSION MATRIX
# =========================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:\n")

print(cm)

# =========================================
# VISUALIZE CONFUSION MATRIX
# =========================================

plt.figure(figsize=(6,5))

sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues",

    xticklabels=class_names,

    yticklabels=class_names
)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Skin Cancer Confusion Matrix")

plt.show()