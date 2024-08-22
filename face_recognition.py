import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

########## KNN CODE ############
def distance(v1, v2):
	# Eucledian 
	return np.sqrt(((v1-v2)**2).sum())
 
def knn(train, test, k=5):
	dist = []
	
	for i in range(train.shape[0]):
		# Lấy vector và nhãn
		ix = train[i, :-1]
		iy = train[i, -1]
		# Tính khoảng cách từ điểm kiểm tra
		d = distance(test, ix)
		dist.append([d, iy])
	# Sắp xếp theo khoảng cách và lấy k hàng đầu
	dk = sorted(dist, key=lambda x: x[0])[:k]
	# Chỉ lấy nhãn
	labels = np.array(dk)[:, -1]
	
	# Lấy tần số của mỗi nhãn
	output = np.unique(labels, return_counts=True)
	# Tìm tần số tối đa và nhãn tương ứng
	index = np.argmax(output[1])
	return output[0][index]


cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

dataset_path = "./face_dataset/"

face_data = []
labels = []
class_id = 0
names = {}
# Biến lưu trữ các chỉ số hiệu quả
accuracy_list = []
recall_list = []
precision_list = []
f1_score_list = []
processing_time_list = []

# Chuẩn bị bộ dữ liệu
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		names[class_id] = fx[:-4]
		data_item = np.load(dataset_path + fx)
		face_data.append(data_item)

		target = class_id * np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_labels = np.concatenate(labels, axis=0).reshape((-1, 1))
print(face_labels.shape)
print(face_dataset.shape)

trainset = np.concatenate((face_dataset, face_labels), axis=1)
print(trainset.shape)

font = cv2.FONT_HERSHEY_SIMPLEX
count = 0
correct = 0

while True:
	ret, frame = cap.read()
	if ret == False:
		continue
	# Convert frame to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect multi faces in the image
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)

	for face in faces:
		x, y, w, h = face

		# Get the face ROI
		offset = 5
		face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
		face_section = cv2.resize(face_section, (100, 100))

		# Đo thời gian xử lý
		start_time = cv2.getTickCount()
		out = knn(trainset, face_section.flatten())
		end_time = cv2.getTickCount()
		time_elapsed = (end_time - start_time) / cv2.getTickFrequency()
		processing_time_list.append(time_elapsed)

		# Draw rectangle in the original image
		cv2.putText(frame, names[int(out)],(x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,255), 2)

		count += 1
		#  Nhận diện chính xác
		if int(out) == int(face_labels[0][0]): 
			correct += 1

		#Tính toán độ chính xác
		accuracy = correct / count
		accuracy_list.append(accuracy)

		# Tính toán độ nhạy
		if count > 0:
			recall = correct / len(face_labels)
			recall_list.append(recall)
		else:
			recall_list.append(0)

		# Tính toán độ đặc hiệu
		if correct > 0:
			precision = correct / count
			precision_list.append(precision)
			# Tính toán F1-score
			if precision > 0 and recall > 0:
				f1_score = 2 * (precision * recall) / (precision + recall)
				f1_score_list.append(f1_score)
			else:
				f1_score_list.append(0)
		else:
			precision_list.append(0)
			f1_score_list.append(0)

		print("Accuracy: {}".format(accuracy))
		print("Recall: {}".format(recall))
		# print("Precision: {}".format(precision))
		# print("F1-score: {}".format(f1_score))
		print("Processing time: {}".format(time_elapsed))

	cv2.imshow("Faces", frame)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

#  Hiển thị biểu đồ thống kê
plt.figure(figsize=(10, 6))

#  Hiển thị độ chính xác
plt.subplot(2, 2, 1)
plt.plot(accuracy_list)
plt.title("Accuracy")
plt.xlabel("Number of detections")
plt.ylabel("Accuracy")

#  Hiển thị độ nhạy
plt.subplot(2, 2, 2)
plt.plot(recall_list)
plt.title("Recall")
plt.xlabel("Number of detections")
plt.ylabel("Recall")

#  Hiển thị độ đặc hiệu
plt.subplot(2, 2, 3)
plt.plot(precision_list)
plt.title("Precision")
plt.xlabel("Number of detections")
plt.ylabel("Precision")

#  Hiển thị F1-score
plt.subplot(2, 2, 4)
plt.plot(f1_score_list)
plt.title("F1-score")
plt.xlabel("Number of detections")
plt.ylabel("F1-score")

plt.tight_layout()
plt.show()

# Hiển thị thời gian xử lý
plt.figure(figsize=(10, 4))
plt.plot(processing_time_list)
plt.title("Processing Time")
plt.xlabel("Number of detections")
plt.ylabel("Processing Time (seconds)")
plt.show()