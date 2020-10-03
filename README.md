## Description
This repository contains face recognition service: ML scripts & API to use them.

## ML Pipeline
ML pipeline contains four main steps:
- face detection to crop faces from images (RetinaFace is used);
- face alignment to descrease the distribution variance; 
- feature extraction ([InsightFace](https://github.com/deepinsight/insightface) is used);
- kNN based on cosine distance on top of feature vectors. 

ML pipeline is encapsulated into Index class which provides two main methods: `add` and `find`. 
Index can be used in two ways: eyes-only (works better with masked people) and full faces (works better with open faces).

(See [face/](https://github.com/feanor-on-fire/face_recognition/tree/master/face) for details). 

### Known issues: 
- add CUDA support for inference; 
- adapt alignment algorithm from InsightFace instead of own implementation to decrease distribution shift; 
- nearest neighbours algoritm is full-scan (O(n) complexity, needs to be replaced with approximate NN search, e.g. `nmslib`).

## API / Wrappers 
API is written with Tornado framework.
End-points:
- /analyze_face - handle post request, that contains image with human face/faces as payload. Return back results of recognition by classifier with extracted embeddings.
