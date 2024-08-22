# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# def evaluate_model(y_true, y_pred, labels):
#     # Tính toán các chỉ số hiệu suất
#     accuracy = accuracy_score(y_true, y_pred)
#     precision = precision_score(y_true, y_pred, average='weighted')
#     recall = recall_score(y_true, y_pred, average='weighted')
#     f1 = f1_score(y_true, y_pred, average='weighted')

#     print(f"Accuracy: {accuracy:.2f}")
#     print(f"Precision: {precision:.2f}")
#     print(f"Recall: {recall:.2f}")
#     print(f"F1-Score: {f1:.2f}")

#     # Vẽ sơ đồ hiệu suất
#     metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-Score': f1}
#     plt.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'red', 'orange'])
#     plt.title('Model Performance Metrics')
#     plt.ylim(0, 1)
#     plt.ylabel('Score')
#     plt.show()

# def plot_confusion_matrix(y_true, y_pred, labels):
#     cm = confusion_matrix(y_true, y_pred, labels=labels)
#     plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
#     plt.title('Confusion Matrix')
#     plt.colorbar()
#     tick_marks = np.arange(len(labels))
#     plt.xticks(tick_marks, labels, rotation=45)
#     plt.yticks(tick_marks, labels)
    
#     # Ghi số vào từng ô trong ma trận
#     for i in range(len(labels)):
#         for j in range(len(labels)):
#             plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.tight_layout()
#     plt.show()

# # Giả sử bạn đã có dữ liệu thực tế và dự đoán
# y_true = ['person1', 'person2', 'person1', 'person3', 'person2']  # Dữ liệu nhãn thật
# y_pred = ['person1', 'person2', 'person3', 'person3', 'person2']  # Dự đoán từ model
# labels = ['person1', 'person2', 'person3']  # Các nhãn có thể có

# # Gọi hàm đánh giá
# evaluate_model(y_true, y_pred, labels)
# plot_confusion_matrix(y_true, y_pred, labels)
