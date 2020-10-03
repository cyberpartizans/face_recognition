import os
from logging import getLogger

import cv2
import insightface
import joblib as jl
import numpy as np
import torch
from retinaface.pre_trained_models import get_model
from torchvision import transforms

from face.align import FaceAligner

logger = getLogger(__name__)


class FacePredictor:
    def __init__(self):
        self.model = get_model("resnet50_2020-07-20", max_size=1024)
        self.model.eval()
        # ToDo: move to CUDA

    def __call__(self, x):
        return self.model.predict_jsons(x)


class Embedder:
    """
    This class extract features from faces.
    """

    def __init__(self):
        embedder = insightface.iresnet100(pretrained=True)
        embedder.eval()
        self.model = embedder
        # ToDo: move to CUDA

        mean = [0.5] * 3
        std = [0.5 * 256 / 255] * 3

        self.preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    def __call__(self, face: np.ndarray):
        """
        :param face: np.array with uint8 BGR image, 112*112
        :return: feature vector, dim=512
        """
        assert 1 < face.max() < 256
        face = face[:, :, ::-1]  # BGR -> RGB
        tensor = self.preprocess(face.astype('float32') / 255.)

        with torch.no_grad():
            features, = self.model(tensor.unsqueeze(0))
        return features


class Index:
    def __init__(self, use_eye_area=True, state_path=None, storage_dir='.'):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)
        self.face_detector = FacePredictor()
        self.face_aligner = FaceAligner(desiredFaceWidth=112)
        self.feature_extractor = Embedder()
        self.features = {}
        if state_path is not None:
            self.features = jl.load(state_path)

        mask = np.zeros((112, 112, 3), dtype=float)
        if use_eye_area:
            mask[8:48, 8:-8, :] = 1
        else:
            mask[...] = 1
        self.mask = mask

    def get_from_storage(self, name):
        img = cv2.imread(os.path.join(self.storage_dir, f'{name}.jpg'))
        assert img is not None, f"can not find {name}"
        return img

    def make_image(self, ref, img, masked, score):
        res = np.hstack((ref, img, masked))
        res = cv2.putText(res, f'score = {score:.2f}', (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
        return res

    def find(self, image, top_k=10):
        # ToDo: full scan if uneffective, switch to nmslib
        keys, values = zip(*self.features.items())
        res = []
        values = torch.stack(values, dim=0)
        for warped, masked in self.extract_faces(image):
            features = self.feature_extractor(masked)
            scores = torch.cosine_similarity(features.view(1, -1), values, dim=1)
            best_matches = torch.argsort(scores)[-top_k:]
            for match in best_matches:
                name = keys[match]
                dispayed_image = self.make_image(ref=warped, img=self.get_from_storage(name),
                                                  score=scores[match], masked=masked)
                res.append({'image': dispayed_image,
                            'score': scores[match],
                            'index': match,
                            })
        return res

    def add(self, image, name):
        faces = self.extract_faces(image)
        for i, (warped, masked) in enumerate(faces):
            subname = f'{i}_{name}'
            features = self.feature_extractor(masked)
            cv2.imwrite(os.path.join(self.storage_dir, f'{subname}.jpg'), warped)
            self.features[subname] = features

    def save_state(self, path='state.bin'):
        logger.info(f'Index state has been saved to {path}')
        jl.dump(self.features, path)

    def extract_faces(self, img):
        min_size = 80
        pred = self.face_detector(img)
        logger.info(f"{len(pred)} faces extracted")
        for i, ann in enumerate(pred):
            if ann["score"] < .5:
                continue
            x_min, y_min, x_max, y_max = ann["bbox"]
            bbox = img[y_min: y_max, x_min: x_max, :]

            h, w, _ = bbox.shape
            if h < min_size or w < min_size:
                continue

            landmarks = ann["landmarks"]
            landmarks -= np.array([x_min, y_min])
            right, left, *_ = landmarks
            warped = self.face_aligner.align(bbox, left_eye=left, right_eye=right)
            masked = warped * self.mask
            yield warped, masked
