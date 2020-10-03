from tempfile import mkdtemp

import cv2
import pytest

from face.index import Index


@pytest.mark.parametrize(
    'use_eyes_only',
    (True, False)
)
def test_index(use_eyes_only):
    lena = cv2.imread('face/fixtures/lena.png')
    kranik1 = cv2.imread('face/fixtures/karanik.jpg')
    kranik2 = cv2.imread('face/fixtures/karanik2.png')

    dirname = mkdtemp()

    index = Index(storage_dir=dirname, use_eye_area=use_eyes_only)
    index.add(lena, 'lena')
    index.add(kranik1, 'karanik')
    assert len(index.features) == 2

    bad_match, good_match = index.find(kranik2, top_k=2)
    assert good_match['index'] == 1
    assert good_match['score'] > .7
    assert bad_match['score'] < .3
