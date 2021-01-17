import numpy as np
import cv2
import os
import imutils
from tensorflow.keras.models import load_model



#Path to prototxtfile
prototxt_file = "face_detector//deploy.prototxt"
#Path to caffe_model
caffeModel = "face_detector//res10_300x300_ssd_iter_140000.caffemodel"


# Loading my encoder and decoder
enc = load_model(".//ModelCheckpoints//VAE_GANS_encoder.h5")
dec = load_model(".//ModelCheckpoints//VAE_GANS_decoder.h5")


face_detector = cv2.dnn.readNet(prototxt_file, caffeModel)

# Detect faces
def detect_faces(frame, face_detector):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (400, 400), (104.0, 177.0, 123.0))

    face_detector.setInput(blob)
    detections = face_detector.forward()

    locs = []; confidences = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7]*np.array([w, h, w, h])
            (startx, starty, endx, endy) = box.astype("int")

            (startx, starty) = (max(0, startx), max(0, starty))
            (endx, endy) = (min(w-1, endx), min(h-1, endy))

            locs.append((startx, starty, endx, endy))
            confidences.append(confidence)

    locs = np.array(locs)
    confidences = np.array(confidences, dtype="int")
    # print(f"data type of : {type(confidences[0])}")
    
    return locs


def send_data(face):
    test_data = []
    # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (128, 128))
    face = cv2.GaussianBlur(face, (3, 3), 0)
    test_data.append(np.expand_dims(face, axis = 2))
    test_data = np.array(test_data, dtype='float32')
    test_data = test_data/255.
    z_val_mean, z_val_log_var, z_val = enc(test_data)
    decoded_val_imgs = dec.predict(z_val)
    decode_img = np.squeeze(decoded_val_imgs[0]*255, axis = 2)
    decode_img = decode_img.astype('uint8')
    # decode_img = cv2.resize(decode_img, face_shape)
    return decode_img




#Load the Camera to save faces
vcap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = vcap.read()
    if ret:    
        frame = imutils.resize(frame, width=400)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # orig_gray_frame = gray_frame.copy()
        boxes = detect_faces(frame, face_detector)
        # print(boxes)
        largest_box = 0   
        for box in boxes:
            if box is None:
                continue
            (startX, startY, endX, endY) = box
            # cv2.rectangle(gray_frame, (startX, startY), (endX, endY), (2, 220, 10), 2)                        
            face = gray_frame[startY:endY, startX:endX]
            # face_shape = face.shape
            # gray_frame[startY:endY, startX:endX] = np.zeros(shape=face.shape)
            orig_face = face.copy()
            try:
                print("inside try")
                decode_face = send_data(face)      
                decode_face = cv2.resize(decode_face, face.shape[::-1])          
                
                # decode_face = cv2.rotate(decode_face, cv2.ROTATE_90_CLOCKWISE)
                # print(f"Shape of decode_face : {decode_face.shape}")
                # print(f"shape of face :{face.shape}")
                gray_frame[startY:endY, startX:endX] = decode_face
                decode_frame = cv2.resize(decode_face, (400, 400))
            except:
                continue
            
            orig_frame = cv2.resize(orig_face, (400, 400))
        cv2.imshow("VAE_o/p", decode_frame)
        cv2.imshow('Original', orig_frame)        
        key = cv2.waitKey(1) & 0xFF
        

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        
cv2.destroyAllWindows()
vcap.release()