import cv2
import numpy as np 

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")

while True:
	ret,frame = cap.read()

	gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(gray_frame,1.3,5)
	if len(faces) == 0:
		continue

	for face in faces[:1]:
		x,y,w,h = face

		offset = 10

	# Lấy kích thước của frame
	frame_height, frame_width = frame.shape[:2]

	# Tính toán các tọa độ với offset
	y1 = max(0, y - offset)
	y2 = min(frame_height, y + h + offset)
	x1 = max(0, x - offset)
	x2 = min(frame_width, x + w + offset)

	# Đảm bảo các tọa độ nằm trong giới hạn của frame
	if y1 < y2 and x1 < x2:
		face_offset = frame[y1:y2, x1:x2]
		face_selection = cv2.resize(face_offset, (100, 100))
		cv2.imshow("Face", face_selection)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	cv2.imshow("faces", frame)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()