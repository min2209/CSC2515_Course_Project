import numpy as np
import scipy.io as scio

# loads .mat file at <path> into dictionary.
# access dictionary content by mat_content["<variable_name>"]

def load_mat(path):
    mat_content = scio.loadmat(path)
    return mat_content


if __name__ == "__main__":
    mat_contents = load_mat("/media/min/Data/SVHN/Format2/test_32x32.mat")
    X = mat_contents["X"]
    img1 = X[:,:,1,1]
    pass