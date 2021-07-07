import os
from typing import AnyStr, List, overload

import cv2
import numpy as np

IMAGE_DIR = 'train_img'
EDGE_DIR = 'train_edge_all/train_edge'
COMPARE_DIR = 'VITON/VITON-men/train_color'
# COPY_DIR = 'VITON/VITON-men/train_edge'
COPY_DIR = 'VITON/VITON-men/train_color_submit'
DATASET_TRAIN_COLOR = 'VITON/VITON-men/train_color'
DATASET_TRAIN_EDGE = 'VITON/VITON-men/train_edge'


def copy(f1, f2):
    original_edge = f'{EDGE_DIR}/{f1}'
    copy_edge = f'{COPY_DIR}/{f2}'
    if not os.path.exists(copy_edge):
        os.system(f'cp "{original_edge}" "{copy_edge}"')


def rename_finished_edges():
    if not os.path.exists(COPY_DIR):
        os.mkdir(COPY_DIR)
    images = list(map(lambda x: os.path.join(IMAGE_DIR, x), os.listdir(IMAGE_DIR)))
    compares = list(map(lambda x: os.path.join(COMPARE_DIR, x), os.listdir(COMPARE_DIR)))
    for e, c in enumerate(compares, start=1):
        if os.path.basename(c) in os.listdir(COPY_DIR):
            continue
        print(e, '/', len(compares))
        for i in images:
            img1 = cv2.imread(c)
            img2 = cv2.imread(i)
            try:
                if img1.shape == img2.shape and not np.bitwise_xor(img2, img1).any():
                    copy(os.path.basename(i), os.path.basename(c))
            except Exception as e:
                print(e)


def to_submit():
    if not os.path.exists(COPY_DIR):
        os.mkdir(COPY_DIR)
    for i in list(set(os.listdir(DATASET_TRAIN_COLOR)).difference(os.listdir(DATASET_TRAIN_EDGE))):
        original_edge = f'{DATASET_TRAIN_COLOR}/{i}'
        copy_edge = f'{COPY_DIR}/{i}'
        if not os.path.exists(copy_edge):
            os.system(f'cp "{original_edge}" "{copy_edge}"')


def resize_images(*filenames):
    if filenames is None:
        raise Exception('Files paths are missing!')
    for i in filenames:
        if i is None or not os.path.exists(i):
            raise Exception(f'\'{i}\' is not a valid path.')
        if not i.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            raise Exception(f'\'{i}\' is not an image file.')
        cv2.imwrite(i, cv2.resize(cv2.imread(i), (192, 256)))


list_images = lambda x: [os.path.join(x, i) for i in os.listdir(x)]
resize_images(*list_images('VITON/VITON-men/train_color'),
              *list_images('VITON/VITON-men/train_img'), *list_images('VITON/VITON-men/train_edge'))
