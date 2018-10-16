# Project: Face Recognition System with Deep Learning Models and OpenCV
# Objective: Recognize face based on image file
# Extracted from pyimageserach article: https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/

# Step 0 - Import libraries

# imutils package was created by Adrian Rosebrock with some useful image processing functions
# Repo: github.com/jrosebr1/imutils
from imutils import paths
import imutils

import numpy as np
import argparse
import pickle
import cv2
import os

# Construct argument parser (argparse), using args from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="input image for facial recognition")
ap.add_argument("-t", "--detector", required=True, help="path to OpenCV's deep learning model for face detection")
ap.add_argument("-m", "--model", required=True, help="path to OpenCV's deep learning model for face embedding")
ap.add_argument("-r", "--recognizer", required=True, help="path to model trained for facial recognition")
ap.add_argument("-l", "--le", required=True, help="path to label encoder")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability (confidence) to filter weak detections")
args = vars(ap.parse_args())

# Step 1 - Load Face Detection Model

# load serialized face detector from disk
print("[INFO] load face detector ...")

# deploy.prototxt contains a text description of the deep neural network (DNN) architecture
protoPath = os.path.sep.join([args["detector"], "deploy.prototxt"])

# .caffemodel contains the content of the Caffe model (binary file with trained weights)
# in this case, we are using reduced ResNet-10 model, based on SSD framework (Single Shot Multibox Detector)
modelPath = os.path.sep.join([args["detector"], "res10_300x300_ssd_iter_140000.caffemodel"])

# Reads a DNN stored in Caffe model in memory, using DNN architecture (prototxt - Caffe)
detector = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# Step 2 - Load Face Embedding Model (To generate 128D Vector for face detected)

# load serialized face embedding generator from disk
print("[INFO] load face recognizer (embedding) ...")

# Reads a DNN stored in Torch model in memory
embedder = cv2.dnn.readNetFromTorch(args["model"])

# Step 3 - Load Trained Model for Facial Recognition

# load serialized face recognition trained model 
recognizer = pickle.loads(open(args["recognizer"], "rb").read())

# Step 4 - Load Label Encoder for Facial Recognition

# load serialized face recognition label encoder
le = pickle.loads(open(args["le"], "rb").read())

# Step 4 - Facial Recognition
# Process evolve:
#   1) Prepare image
#   2) Face detection
#   3) Face embeddings
#   4) Predict Name (Label) and probability, using recognizer model (SVM)

# Step 4.1 - Prepare image
print("[INFO] processing image...")

# load input image using OpenCV
image = cv2.imread(args["image"])

# resize image with width of 600px, maintaining aspect ratio, using imutils from Adrian
image = imutils.resize(image, width=600)

# get image dimensions
(h, w) = image.shape[:2]

# Step 4.2 - Face Detection

# Construct a blob from the image, using OpenCV
imageBlob = cv2.dnn.blobFromImage(image=cv2.resize(image, (300,300)), scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)

# Apply DNN for Face Detection in input image. Run forward pass
detector.setInput(imageBlob)
detections = detector.forward()

# Step 4.3 - Face Embeddings

# ensure at least one face was found
if len(detections) > 0:

    # loop over the detections
    for i in range(0, detections.shape[2]):

        # Detector returns a vector with probability in index 2 and bounding box between indexes 3 and 7
        confidence = detections[0, 0, i, 2]

        # Ensure that largest probability also means minimal confidence test (filtering weak detections)
        if confidence > args["confidence"]:
            # compute the (x,y) coordinates of the bounding box for the face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # extract the face ROI (Region of Interest) and get ROi dimensions
            face = image[startY:endY, startX:endX]
            (fH, fW) = face.shape[:2]

            # ensure the face width and height are at least 20x20
            if fW < 20 or fH < 20:
                continue
            
            # contruct a blob for face ROI, then pass blob into embedding model
            # to obtain the 128-d vector of the face
            faceBlob = cv2.dnn.blobFromImage(image=face, scalefactor=1.0 / 255, size=(96, 96), mean=(0, 0, 0), swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()

            # Step 4.4 - Predict Name (Label), using recognizer model (SVM)
            preds = recognizer.predict_proba(vec)[0]

            # Get max probability from prediction
            j = np.argmax(preds)

            # Get probability and label (Person Name)
            proba = preds[j]
            name = le.classes_[j]

            # Step 5 - Draw Rectangle on Face with Predicted Name and Probability
            text = "{}: {:.2f}%".format(name, proba * 100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 2)
            cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 2)

# show the output image, with face recognition
cv2.imshow("Image", image)
cv2.waitKey(0)







