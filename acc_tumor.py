from sklearn.metrics import accuracy_score
import os
from app import predict

y_true = []
y_pred = []

test_folder = "datasets/testing"

for label_name in os.listdir(test_folder):
    class_folder = os.path.join(
        test_folder,
        label_name
    )
    for img_name in os.listdir(class_folder):
        img_path = os.path.join(
            class_folder,
            img_name
        )
        pred, conf, model = predict(
            img_path,
            "brain_tumor"
        )
        y_true.append(label_name)
        y_pred.append(pred)

acc = accuracy_score(
    y_true,
    y_pred
)

print(
    f"Overall Accuracy: {acc*100:.2f}%"
)