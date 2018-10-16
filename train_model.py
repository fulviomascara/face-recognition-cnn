# Project: Face Recognition System with Deep Learning Models and OpenCV
# Objective: Train model for face recognition, using embeddings database
# Extracted from pyimageserach article: https://www.pyimagesearch.com/2018/09/24/opencv-face-recognition/

# Step 0 - Import libraries
# Support Vector Machine and LabelEncoder libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

import argparse
import pickle

# Construct argument parser (argparse), using args from command line
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--embeddings", required=True, help="path to serialized db of facial embeddings (pickle file)")
ap.add_argument("-r", "--recognizer", required=True, help="path to output model trained for facial recognition")
ap.add_argument("-l", "--le", required=True, help="path to output label encoder file")
args = vars(ap.parse_args())

# Step 1 - Encode Labels (Person Names)

# Open Embeddings File
print("[INFO] load face embeddings database ...")
data = pickle.loads(open(args["embeddings"], "rb").read())

# encode labels, using name field on embeddings database
print("[INFO] encoding labels ...")
le = LabelEncoder()
labels = le.fit_transform(data["names"])

# Step 2 - Train model with SVM, using 128-d embedding database, generating actual face recognition
# Remember that SVM only works with 2 or more classes
print("[INFO] training model ...")
recognizer = SVC(C=1.0, kernel="linear", probability=True)
recognizer.fit(data["embeddings"], labels)

# Step 3 - Dump trained model for actual facial recognition
print("[INFO] dump trained model ...")
f = open(args["recognizer"], "wb")
f.write(pickle.dumps(recognizer))
f.close()

# Step 4 - Dump Label Encoder from actual facial recognition
print("[INFO] dump label encoder ...")
f = open(args["le"], "wb")
f.write(pickle.dumps(le))
f.close()