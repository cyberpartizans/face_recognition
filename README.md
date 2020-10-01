## Description
This repository contains face recognition service: ML scripts & API to use them.
## Pipeline
#### 1. Face detection
First step is detect human face on image and crop it.
Face detection is implemented with open-source library MTCNN. 
More details in projects repository https://github.com/ipazc/mtcnn
#### 2. Feature extractor
After face detection whe need to extract face embedding. 
In this projects we use pre-trained FaceNet by Hiroki Taniai described in this repository https://github.com/nyoki-mtl/keras-facenet
#### 3. Classifier (IN PROGRESS)
Final step is use SVM classifier with gotten embeddings.
#### 4. API
API is written with Tornado framework.
End-points:
- /analyze_face - handle post request, that contains image with human face/faces as payload. Return back results of recognition by classifier with extracted embeddings.