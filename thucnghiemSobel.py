import cv2
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, roc_curve, auc
import matplotlib.pyplot as plt
import scipy.io as sio

# Hàm tính toán độ đo precision, recall, và F1-score
def evaluate_edge_detection(ground_truth, detected_edges):
    precision, recall, f1, _ = precision_recall_fscore_support(ground_truth, detected_edges, average='binary')
    return precision, recall, f1

# Hàm hiển thị ROC Curve và tính AUC
def plot_roc_curve(ground_truth, detected_edges):
    fpr, tpr, _ = roc_curve(ground_truth, detected_edges)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    return roc_auc

# Hàm phát hiện cạnh Sobel
def sobel_edge_detection(image, threshold=0.5):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)
    edges = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    edges /= np.max(edges)
    binary_edges = (edges > threshold).astype(np.uint8)
    return binary_edges


# Ví dụ thực hiện thuật toán phát hiện cạnh và đánh giá trên một ảnh
def perform_experiment(image_path, ground_truth_path, threshold=0.5):
    image = cv2.imread(image_path)
    
    # Load ground truth từ file .mat
    mat_contents = sio.loadmat(ground_truth_path)
    ground_truth = mat_contents['groundTruth'][0][0][0][0][1]  # Chọn trường chứa ground truth phù hợp
    ground_truth = ground_truth.astype(np.uint8)
    
    # Áp dụng thuật toán Sobel
    detected_edges = sobel_edge_detection(image, threshold)
    
    # Đánh giá
    precision, recall, f1 = evaluate_edge_detection(ground_truth.flatten(), detected_edges.flatten())
    roc_auc = plot_roc_curve(ground_truth.flatten(), detected_edges.flatten())
    
    return precision, recall, f1, roc_auc


# Thực hiện thí nghiệm với một ảnh và chú thích cạnh
image_path = os.path.join('images/test/5096.jpg') # thay đổi đường dẫn để chọn ảnh khác
ground_truth_path = os.path.join('ground_truth/test/5096.mat')
precision, recall, f1, roc_auc = perform_experiment(image_path, ground_truth_path)

print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1-Score: {f1:.4f}')
print(f'AUC: {roc_auc:.4f}')
