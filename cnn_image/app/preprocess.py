"""
    Archivo: preprocess.py
    Propósito: Funciones para aplicar diferentes operaciones de preprocesamiento a una imagen.
    Autor: Victor Villarreal, Cristian Garcia
    Fecha: 2025-11-18
"""
import cv2
import numpy as np
from config import KERNEL_EDGES, KERNEL_SHARPEN


def to_rgb_array(pil_img) -> np.ndarray:
    img = np.array(pil_img.convert('RGB'))
    return img

def smooth(img: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(img, (5,5), 0)

def edges(img: np.ndarray) -> np.ndarray:
    kernel = KERNEL_EDGES
    return cv2.filter2D(img, -1, kernel)

def sharpen(img: np.ndarray) -> np.ndarray:
    kernel = KERNEL_SHARPEN
    return cv2.filter2D(img, -1, kernel)

def resize_and_norm(img: np.ndarray, size=(64,64)) -> np.ndarray:
    img = cv2.resize(img, np.size)
    img = img.astype('float32')/255.0
    # transpose to C,H,W for PyTorch
    img = np.transpose(img, (2,0,1))
    return img