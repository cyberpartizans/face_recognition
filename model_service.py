import json
import argparse
from json import loads
from io import BytesIO
from typing import Optional, Awaitable

import requests
from tornado.ioloop import IOLoop
from tornado.httpserver import HTTPServer
from tornado.web import RequestHandler, Application

from PIL import Image
from tensorflow.keras.models import load_model

from get_features import get_face_features
from face_comparission import search_face

params_parser = argparse.ArgumentParser(description='Process some integers.')
params_parser.add_argument('--port')


class FaceHandler(RequestHandler):
    model = load_model('facenet_keras.h5')

    def __init__(self, application, request, **kwargs):
        super().__init__(application, request, **kwargs)

    def post(self, *args, **kwargs):
        data = json.loads(self.request.body)
        image = self.__load_image_from_url(data['Url'])
        face_features = get_face_features(model=self.model, image=image)
        search_result = search_face(face_features)
        search_result = self.format_response(search_result)
        self.write(search_result)

    def format_response(self, search_result):
        pass

    @staticmethod
    def __load_image_from_url(url):
        url = loads(url)['Url']
        return Image.open(BytesIO(requests.get(url).content)).convert('RGB')

    @staticmethod
    def __load_image_from_bytes(img_bytes):
        return Image.open(BytesIO(img_bytes)).convert('RGB')

    def data_received(self, chunk: bytes) -> Optional[Awaitable[None]]:
        pass


def create_app():
    app = Application([
        (r"/analyze_face", FaceHandler),
    ])

    return app


if __name__ == "__main__":
    args = params_parser.parse_args()

    port = int(args.port)
    server = HTTPServer(create_app())
    server.listen(port)
    server.start()  # Forks multiple sub-processes
    print('Face service started at the %d port' % port)
    IOLoop.current().start()
