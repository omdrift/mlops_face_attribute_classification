"""Shared preprocessing; BGR order preserves the original training convention."""
import hashlib

import cv2
import numpy as np

PREPROCESS_VERSION = "white-crop-bgr-64-v1"


def preprocess_image(img, size=64):
    if img is None:
        raise ValueError("Image could not be decoded")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coords = cv2.findNonZero(cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)[1])
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y + h, x:x + w]
    return cv2.resize(img, (size, size)).astype(np.float32) / 255.0


def array_hash(array):
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str(array.shape).encode() + str(array.dtype).encode() + array.tobytes()).hexdigest()
