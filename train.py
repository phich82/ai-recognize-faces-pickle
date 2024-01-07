from architecture import *
import os
import cv2
import mtcnn
import pickle
import numpy as np
from sklearn.preprocessing import Normalizer
from tensorflow.keras.models import load_model
import datetime
import shutil
import time


def normalize(img):
    mean, std = img.mean(), img.std()
    return (img - mean) / std

def encodings(
    train_path: str=None,
    face_detector=None,
    face_encoder=None,
    l2_normalizer=None,
    required_shape: tuple=(160, 160)
):
    encodes = []
    encoding_dict = dict()

    for face_name in os.listdir(train_path):
        person_dir = os.path.join(train_path, face_name)

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)

            img_BGR = cv2.imread(image_path)
            img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)

            x = face_detector.detect_faces(img_RGB)
            x1, y1, width, height = x[0]['box']
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1+width, y1+height
            face = img_RGB[y1:y2, x1:x2]

            face = normalize(face)
            face = cv2.resize(face, required_shape)
            face_d = np.expand_dims(face, axis=0)
            encode = face_encoder.predict(face_d)[0]
            encodes.append(encode)

        if encodes:
            encode = np.sum(encodes, axis=0)
            encode = l2_normalizer.transform(np.expand_dims(encode, axis=0))[0]
            encoding_dict[face_name] = encode

    return encoding_dict

def train(
    train_path,
    model_path: str=None,
    face_detector=None,
    face_encoder=None,
    l2_normalizer=None,
    required_shape: tuple=(160, 160),
    out_dir: str='runs'
):

    encoding_dict = encodings(
        train_path=train_path,
        face_detector=face_detector,
        face_encoder=face_encoder,
        l2_normalizer=l2_normalizer,
        required_shape=required_shape
    )

    timestamp = str(int(datetime.datetime.now().timestamp()))
    out_dir = f'{out_dir}/{timestamp}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, mode=0o777, exist_ok=True)

    filename = 'encodings.pkl'
    if os.path.exists(model_path):
        print(f'copying => {model_path}')
        shutil.copy(model_path, out_dir)
        print('copy successful')
        filename = model_path.split('/').pop()

    encodings_model_file = f'{out_dir}/{filename}'
    with open(encodings_model_file, 'wb') as file:
        pickle.dump(encoding_dict, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'New model created at: {encodings_model_file}')

if __name__ == '__main__':
    face_data = 'data/faces/train'
    facenet_weight_path = 'models/pretrained/facenet_keras_weights.h5'
    model_path = 'models/weights/encodings.pkl'
    required_shape = (160, 160)
    face_encoder = InceptionResNetV2()
    face_encoder.load_weights(facenet_weight_path)
    face_detector = mtcnn.MTCNN()
    l2_normalizer = Normalizer('l2')

    print('Starting to train...')

    start = time.time() # seconds

    train(
        train_path=face_data,
        model_path=model_path,
        face_detector=face_detector,
        face_encoder=face_encoder,
        l2_normalizer=l2_normalizer,
        required_shape=required_shape,
        out_dir='runs'
    )

    end = time.time()
    print('Time total: ', f'{end - start} seconds')

    print('Training successful')
