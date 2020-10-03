# Face Index

Face Index is a naive realization of end-to-end pipeline to find similar faces.

Face Index is not trainable atm and reuses existing models for face detection and feature extraction.

Usage example:
```python
import os
from glob import glob

import cv2
from tqdm import tqdm

from index import Index

index = Index()
images_pattern = '../images/*.jpg'

for f in tqdm(glob(images_pattern)):
    name, _ = os.path.basename(f).split('.')
    img = cv2.imread(f)
    index.add(img, name)

img = cv2.imread('face_to_find.jpg')
good_match,  = index.find(iimg, top_k=1)
``` 