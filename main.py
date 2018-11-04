# FILENAME: main.py
# AUTHORS: Jalocha, Kozlowski, Piekarski

import cv2
import numpy as np

def read_image(filepath):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)


def calculate_target_fcn(in_image_path, ref_image_path):
    in_image = read_image(in_image_path)
    ref_image = read_image(ref_image_path)
    TP = TN =  FP = FN = 0
    for i in range(0, ref_image.shape[0]):
        for j in range(0, ref_image.shape[1]):
            if ref_image[i][j] != 0 and in_image[i][j] != 0:
                TP += 1
            elif ref_image[i][j] != 0 and in_image[i][j] == 0:
                FN += 1
            elif ref_image[i][j] == 0 and in_image[i][j] == 0:
                TN += 1
            elif ref_image[i][j] == 0 and in_image[i][j] != 0:
                FP += 1
    return TP/(TP + FP + FN)


def calculate_pix_variance(image):
    variance_matrix = np.zeros((image.shape[0], image.shape[1]))
    for i in range(1, image.shape[0]-1):  # edges to be considered later
        for j in range(1, image.shape[1]-1):
            var1 = np.abs(int(image[i+1][j]) - int(image[i-1][j]))
            var2 = np.abs(int(image[i][j+1]) - int(image[i][j-1]))
            var3 = np.abs(int(image[i-1][j-1]) - int(image[i+1][j+1]))
            var4 = np.abs(int(image[i-1][j+1]) - int(image[i+1][j-1]))
            variance_matrix[i][j] = var1 + var2 + var3 + var4
    return variance_matrix


if __name__ == "__main__":
    in_image_path = 'input_data/house_prewitt.png'
    ref_image_path = 'input_data/house_prewitt.png'
    img = read_image(ref_image_path)
    cv2.imshow('test_window', img)
    cv2.waitKey(0)
    print(calculate_target_fcn(in_image_path, ref_image_path))
    print(calculate_pix_variance(img))
