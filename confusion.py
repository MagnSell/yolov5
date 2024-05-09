
# import supervision as sv
# from ultralytics import YOLO
# import numpy as np

# dataset = sv.DetectionDataset.from_yolo("Aquarium-Fish-1/test/images", "Aquarium-Fish-1/test/labels","data.yaml")

# model = YOLO("best.pt")
# def callback(image: np.ndarray) -> sv.Detections:
#     result = model(image)[0]
#     return sv.Detections.from_ultralytics(result)

# confusion_matrix = sv.ConfusionMatrix.benchmark(
#    dataset = dataset,
#    callback = callback
# )

# confusion_matrix.plot()
#class x_center y_center width height
import numpy as np
from sklearn.metrics import confusion_matrix
import os
# Load the true labels from file
# Load all labels from the folder
label_folder = "Aquarium-Fish-1/test/labels"
label_files = os.listdir(label_folder)

# Load all predicted labels from the folder
pred_folder = "runs/detect/exp08/labels"
pred_files = os.listdir(pred_folder)

TP = 0
FP = 0
TN = 0
FN = 0

for i in range(len(pred_files)):
    #Comment in when working on 08
    if i >10:
        label_file = label_files[i+1]
    else:
        label_file = label_files[i]
    #Comment in when working on not 08
    #label_file = label_files[i]
    pred_file = pred_files[i]
    true_labels = np.loadtxt(os.path.join(label_folder, label_file), dtype=float)
    pred_labels = np.loadtxt(os.path.join(pred_folder, pred_file), dtype=float)
    print(label_file)
    print(pred_file)
    if pred_labels.ndim > 1 and true_labels.ndim > 1:
        for pred_label in pred_labels:
            found_match = False
            for true_label in true_labels:
                if abs(pred_label[1] - true_label[1]) < 0.1 and abs(pred_label[2] - true_label[2]) < 0.1:
                    found_match = True
                    break
            if found_match:
                TP += 1
            else:
                FP += 1

        for true_label in true_labels:
            found_match = False
            for pred_label in pred_labels:
                if abs(pred_label[1] - true_label[1]) < 0.1 and abs(pred_label[2] - true_label[2]) < 0.1:
                    found_match = True
                    break
            if not found_match:
                FN += 1
    if pred_labels.ndim == 1 and true_labels.ndim == 1:
        if abs(pred_labels[1] - true_labels[1]) < 0.1 and abs(pred_labels[2] - true_labels[2]) < 0.1:
            TP += 1
        else:
            FP += 1
            FN += 1
    if pred_labels.ndim > 1 and true_labels.ndim == 1:
        for pred_label in pred_labels:
            if abs(pred_label[1] - true_labels[1]) < 0.1 and abs(pred_label[2] - true_labels[2]) < 0.1:
                TP += 1
            else:
                FP += 1

    if pred_labels.ndim == 1 and true_labels.ndim > 1:
        for true_label in true_labels:
            if abs(pred_labels[1] - true_labels[1]) < 0.1 and abs(pred_labels[2] - true_labels[2]) < 0.1:
                TP += 1
            else:
                FN += 1

TN = 0
# Comment in when working on 08
FN += 3


confusion_matrix = np.array([[TP, FP], [FN, TN]])
import matplotlib.pyplot as plt

# Display the confusion matrix
plt.text(0, 0, TP, ha='center', va='center', color='black')
plt.text(1, 0, FP, ha='center', va='center', color='black')
plt.text(0, 1, FN, ha='center', va='center', color='black')
plt.text(1, 1, TN, ha='center', va='center', color='black')
plt.imshow(confusion_matrix, cmap='Blues')
plt.title('Confusion Matrix')
plt.colorbar()
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.xticks([0, 1], ['Fish', 'Environment'])
plt.yticks([0, 1], ['Fish', 'Environment'])
plt.show()
