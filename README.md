# Facial Recognition using Deep Learning and OpenCV

Motivated by the challenge declared by the innovation committee and inspired by lessons from Adrian Rosebrock (pyimagesearch.com) about Computer Vision, I started a journey into learning and understanding some fundamentals and approaches about Computer Vision, OpenCV and Deep Learning, which I'm sharing here and expect to help you in your own challenges.

## Challenge Statement

Using a low-cost equipment like Raspberry Pi, I'm on mission to deliver a efficient and reliable facial recognition system, capable to preprocess (detect faces, generate embeddings, train/enrich data) and recognize employees' faces, register events when faces are recognized and finally ensure that certain  resources only can be accessed by certain employees recognized by facial recognition system.

## Setup for MacOS

- python3 (brew install python3)
- opencv >= 3.4 (pip install opencv-contrib-python)
- imutils library (pip install imutils)
- numpy (pip install numpy)
- scikit-learn (pip install scikit-learn)
- Reduced ResNet-10 for Face Detection (face_detection_model folder)
- DNN Architecture for Face Detection (face_detection_model folder)
- OpenFace Trained Model for Embeddings (root folder)

## Components
- extract_embeddings: Create a 128-D Vector representation of faces from image dataset
- train_model: Using actual embeddings, train a model for facial recognition
- recognize: Apply a SVM to recognize faces, using input image and trained model 

## Next Steps
- Transform process steps into functions, for code reuse
- Implement facial recognition in video capture
- Test other Face Detection DNNs / DNN Architectures
- Test other models for facial embeddings
- Test other classification algorithms
- Expose APIs