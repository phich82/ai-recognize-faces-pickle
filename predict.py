import os
import cv2
import numpy as np
import mtcnn
from architecture import *
from train_v2 import normalize, l2_normalizer
from scipy.spatial.distance import cosine
from tensorflow.keras.models import load_model
import pickle
import datetime


CONFIDENCE_THRESH = 0.99
DISTANCE_THRESH = 0.5
REQUIRED_SIZE = (160, 160)

def get_face(img, box):
    x1, y1, width, height = box
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = img[y1:y2, x1:x2]
    return face, (x1, y1), (x2, y2)

def get_encode(face_encoder, face, size):
    face = normalize(face)
    face = cv2.resize(face, size)
    encode = face_encoder.predict(np.expand_dims(face, axis=0))[0]
    return encode


def load_pickle(path):
    encoding_dict = dict()
    with open(path, 'rb') as f:
        encoding_dict = pickle.load(f)
    return encoding_dict

def detect(img, detector=None, encoder=None):
    detector = mtcnn.MTCNN() if detector == None else detector
    encoder = InceptionResNetV2() if encoder == None else encoder
    encoder.load_weights("models/pretrained/facenet_keras_weights.h5")
    encodings = load_pickle('runs/1704635313/encodings.pkl')

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    result = {
        'success': False,
        'detail': None
    }

    for res in results:
        if res['confidence'] < CONFIDENCE_THRESH:
            continue
        face, pt_1, pt_2 = get_face(img_rgb, res['box'])
        encode = get_encode(encoder, face, REQUIRED_SIZE)
        encode = l2_normalizer.transform(encode.reshape(1, -1))[0]
        name = 'unknown'

        distance = float("inf")
        for db_name, db_encode in encodings.items():
            _distance = cosine(db_encode, encode)
            print('_distance => ', _distance)
            if _distance < DISTANCE_THRESH and _distance < distance:
                name = db_name
                distance = _distance
                result.update({
                    'success': True,
                    'detail': {
                        'confidence': res['confidence'],
                        'distance': distance,
                        'name': name
                    }
                })

        if name == 'unknown':
            cv2.rectangle(img, pt_1, pt_2, (0, 0, 255), 2)
            cv2.putText(img, name, pt_1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        else:
            cv2.rectangle(img, pt_1, pt_2, (0, 255, 0), 2)
            cv2.putText(img, name + f'__{distance:.2f}', (pt_1[0], pt_1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 200), 2)
    return result, img

if __name__ == "__main__":
    save_dir = '.out'
    timestamp = str(int(datetime.datetime.now().timestamp()))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    image = cv2.imread('data/tests/Jackie Chan.jpg')
    result, detected_face = detect(image)
    print('result => ', result)
    cv2.imwrite(f"{save_dir}/out_{timestamp}.jpg", detected_face)

