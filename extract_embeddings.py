# Project: Face Recognition System with Deep Learning Models and OpenCV
# Objective: Extract embeddings from images dataset
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
ap.add_argument("-d", "--dataset", required=True, help="path to input directory of faces + images (for training and creating database)")
ap.add_argument("-e", "--embeddings", required=True, help="path to output serialized db of facial embeddings (pickle file)")
ap.add_argument("-t", "--detector", required=True, help="path to OpenCV's deep learning model for face detection")
ap.add_argument("-m", "--model", required=True, help="path to OpenCV's deep learning model for face embedding")
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

# Step 3 - Prepare for Detection and Embedding

# Get Paths to the input images in our dataset
print("[INFO] quantifying faces...")

# Using imutils from Adrian, get full path of every image in dataset folder
imagePaths = list(paths.list_images(args["dataset"]))

# initialize lists for facial embeddings and corresponding names
knownEmbeddings = []
knownNames = []

# initialize Counter of faces processed
total = 0

# Step 4 - Process each image in dataset
# Process evolve:
#   1) Detect name from image path
#   2) Prepare image
#   3) Face detection
#   4) Face embeddings
#   5) Dump embedding and label encoding to file

# Loop over all dataset images
for (i, imagePath) in enumerate(imagePaths):

    # Step 4.1 - Detect name from image path
    print("[INFO] processing image {}/{}".format(i+1, len(imagePaths)))
    personName = imagePath.split(os.path.sep)[-2]

    # Step 4.2 - Prepare image
    
    # load image using OpenCV
    image = cv2.imread(imagePath)

    # resize image with width of 600px, maintaining aspect ratio, using imutils from Adrian
    image = imutils.resize(image, width=600)

    # get image dimensions
    (h, w) = image.shape[:2]

    # Step 4.3 - Face Detection

    # Construct a blob from the image, using OpenCV
    imageBlob = cv2.dnn.blobFromImage(image=cv2.resize(image, (300,300)), scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0), swapRB=False, crop=False)

    # Apply DNN for Face Detection in input image. Run forward pass
    detector.setInput(imageBlob)
    detections = detector.forward()

    # Step 4.4 - Face Embeddings

    # ensure at least one face was found
    if len(detections) > 0:

        # Considering that image has one face, find the bounding box with largest probability calculated by face detection model
        # Detector returns a vector with probability in index 2 and bounding box between indexes 3 and 7
        i = np.argmax(detections[0, 0, :, 2])
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

            # append lists with embeddings and label encoding (Person Name)
            knownNames.append(personName)
            knownEmbeddings.append(vec.flatten())
            total += 1

# Step 4.5 - Dump embedding and label encoding to file
print("[INFO] serializing {} encodings...".format(total))
data = {"embeddings": knownEmbeddings, "names": knownNames}
f = open(args["embeddings"], "wb")
f.write(pickle.dumps(data))
f.close()









