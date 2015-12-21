import numpy as np
import cv2  # Using version 2.4.11

def extract_until_non_none(img, feature_extractor):
    for trim in range(1,(img.shape[0]/2)-1,1):
        desc = feature_extractor(img[trim:-trim, trim:-trim])
        if desc is not None:
            return desc
    return None

def extract_sift(image):
    assert(image.shape[0] == image.shape[1])
    sift = cv2.SIFT()
    img = np.array(image).astype(np.uint8)
    kp, desc = sift.detectAndCompute(img, None)
    return desc

def all_extract_sift(data):
    data_sift = np.array([]).reshape(-1, 128)    # Each example converted into NUM_FEATURES X NUM_DIMENSION
    data_sift_per_example = list()
    count = 1
    duds=[]
    for i in range(len(data)):
        img = data[i]
        count += 1
        if count % 1000 == 0:
            print count
        desc = extract_sift(img)
        if desc is None:
            duds.append(i)
            desc = np.random.random(size=(1, 128))
        data_sift_per_example.append(desc)
        data_sift = np.vstack((data_sift, desc))
    return data_sift, data_sift_per_example, np.array(duds)

def extract_surf(image, hessian=10):
    assert(image.shape[0] == image.shape[1])
    surf = cv2.SURF(hessian)
    #img = np.array(image).astype(np.uint8)
    kp, desc = surf.detectAndCompute(image, None)
    return desc

def all_extract_surf(data):
    data_s = np.array([]).reshape(-1, 128)
    data_per_example = list()
    duds = []
    for i in range(len(data)):
        img = data[i]
        desc = extract_surf(img)
        if desc is None:
            duds.append(i)
            desc = np.random.random(size=(1, 128))
        data_per_example.append(desc)
        data_s = np.vstack((data_s, desc))
    return data_s, data_per_example, np.array(duds)


SZ = 32
affine_flags = cv2.WARP_INVERSE_MAP|cv2.INTER_LINEAR
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img,M,(SZ, SZ),flags=affine_flags)
    return img

def extract_hog(img):
    bin_n = 16
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy)

    # quantizing binvalues in (0...16)
    bins = np.int32(bin_n*ang/(2*np.pi))

    # Divide to 4 sub-squares
    bin_cells = bins[:10,:10], bins[10:,:10], bins[:10,10:], bins[10:,10:]
    mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)
    return hist.reshape(1, -1)

def all_extract_hog(data):
    data_s = np.array([]).reshape(-1, 64)
    data_per_example = list()
    for img in data:
        #img = deskew(img)
        desc = extract_hog(img)
        if desc is None:
            print "Desc is None"
            desc = np.random.random(size=(1, 64))
        data_per_example.append(desc)
        data_s = np.vstack((data_s, desc))
    return data_s, data_per_example

def rgb_to_grayscale(imgs):
    grays = []
    for img in imgs:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        grays.append(gray)

    return np.array(grays)
