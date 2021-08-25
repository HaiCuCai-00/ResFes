# import the necessary packages
# nhập các gói cần thiết
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# grab the dimensions of the frame and then construct a blob
	# from it
    # lấy các kích thước của khung và sau đó tạo một đốm màu
    # từ nó
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))

	# pass the blob through the network and obtain the face detections
    # vượt qua đốm màu qua mạng và nhận được tính năng nhận diện khuôn mặt
	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	# initialize our list of faces, their corresponding locations,
	# and the list of predictions from our face mask network
    # khởi tạo danh sách các khuôn mặt của chúng ta, vị trí tương ứng của chúng,
    # và danh sách các dự đoán từ mạng mặt nạ của chúng tôi
	faces = []
	locs = []
	preds = []

	# loop over the detections
    # vòng lặp qua các phát hiện
	for i in range(0, detections.shape[2]):
		# extract the confidence (i.e., probability) associated with
		# the detection
        # trích xuất độ tin cậy (tức là xác suất) được liên kết với
        # phát hiện
		confidence = detections[0, 0, i, 2]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
        # lọc ra các phát hiện yếu bằng cách đảm bảo độ tin cậy là
        # lớn hơn độ tin cậy tối thiểu
		if confidence > 0.5:
			# compute the (x, y)-coordinates of the bounding box for
			# the object
            # tính tọa độ (x, y) của hộp giới hạn cho
            # đối tượng
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")

			# ensure the bounding boxes fall within the dimensions of
			# the frame
            # đảm bảo các hộp giới hạn nằm trong kích thước của
            # khung
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face ROI, convert it from BGR to RGB channel
			# ordering, resize it to 224x224, and preprocess it
            # trích xuất ROI của khuôn mặt, chuyển đổi nó từ BGR sang kênh RGB
            # đặt hàng, thay đổi kích thước thành 224x224 và xử lý trước
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the face and bounding boxes to their respective
			# lists
            # thêm các hộp mặt và viền vào tương ứng
            # danh sách
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# only make a predictions if at least one face was detected
    # chỉ đưa ra dự đoán nếu ít nhất một khuôn mặt được phát hiện
	if len(faces) > 0:
		# for faster inference we'll make batch predictions on *all*
		# faces at the same time rather than one-by-one predictions
		# in the above `for` loop
        # để suy luận nhanh hơn, chúng tôi sẽ đưa ra dự đoán hàng loạt trên * tất cả *
        # khuôn mặt cùng một lúc thay vì dự đoán từng khuôn mặt
        # trong vòng lặp `for` ở trên
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# return a 2-tuple of the face locations and their corresponding
	# locations
    # trả về 2 bộ vị trí khuôn mặt và vị trí tương ứng của chúng
    # địa điểm
	return (locs, preds)

# load our serialized face detector model from disk
# tải mô hình máy dò khuôn mặt được tuần tự hóa của chúng tôi từ đĩa
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
# tải mô hình phát hiện mặt nạ từ đĩa
maskNet = load_model("mask_detector.model")

# initialize the video stream
# khởi chạy luồng video
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()

# loop over the frames from the video stream
# vòng lặp qua các khung hình từ luồng video
while True:
	# grab the frame from the threaded video stream and resize it
	# to have a maximum width of 400 pixels
    # lấy khung hình từ luồng video theo chuỗi và thay đổi kích thước
    # để có chiều rộng tối đa là 400 pixel
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# detect faces in the frame and determine if they are wearing a
	# face mask or not
    # phát hiện các khuôn mặt trong khung và xác định xem họ có đang đeo
    # mặt nạ hay không
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

	# loop over the detected face locations and their corresponding
	# locations
    # vòng qua các vị trí khuôn mặt được phát hiện và vị trí tương ứng của chúng
    # địa điểm
	for (box, pred) in zip(locs, preds):
		# unpack the bounding box and predictions
        # giải nén hộp giới hạn và dự đoán
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# determine the class label and color we'll use to draw
		# the bounding box and text
        # xác định nhãn lớp và màu mà chúng ta sẽ sử dụng để vẽ
        # hộp giới hạn và văn bản
		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# include the probability in the label
        # bao gồm xác suất trong nhãn
		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# display the label and bounding box rectangle on the output
		# frame
        # hiển thị nhãn và hình chữ nhật hộp giới hạn trên đầu ra
        # khung
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
    # hiển thị khung đầu ra
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key was pressed, break from the loop
    # nếu phím `q` được nhấn, ngắt khỏi vòng lặp
	if key == ord("q"):
		break

# do a bit of cleanup
# dọn dẹp một chút
cv2.destroyAllWindows()
vs.stop()