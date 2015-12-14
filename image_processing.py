import numpy as np
import cv2  # Using version 2.4.11
#from matplotlib.mlab import PCA

def extract_sift(image):#, pca_dimension):
    assert(image.shape[0] == image.shape[1])
    sift = cv2.SIFT()
    img = np.array(image).astype("uint8")
    kp, desc = sift.detectAndCompute(img)
    return desc


def extract_surf(image, hessian=400):
    assert(image.shape[0] == image.shape[1])
    surf = cv2.SURF(hessian)
    kp, desc = surf.detectAndCompute(image)
    return desc


def rgb_to_grayscale(imgs):
    grays = []
    for img in imgs:
        gray = cv2.cvtColor(img.reshape(32,32,3), cv2.COLOR_RGB2GRAY)
        grays.append(gray.reshape(1,-1)[0])

    return np.array(grays)