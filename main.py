# FILENAME: main.py
# AUTHORS: Jalocha, Kozlowski, Piekarski

import cv2
from algorithm import Algorithm


def read_image(filepath):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)


def main():

    in_image_path = 'input_data/house_in.png'
    ref_image_path = 'input_data/house_prewitt.png'
    img_in = read_image(in_image_path)
    img_ref = read_image(ref_image_path)
    algorithm = Algorithm(img_in, img_ref)
    img_edge, pheromone = algorithm.run_algorithm(True)
    print(f'Target fun: {algorithm.model.calculate_target_fcn(img_edge, img_ref)}')
    cv2.imshow('Im edge', img_edge)
    cv2.imshow('Ref', img_ref)
    cv2.imshow('Pheromone', pheromone / pheromone.max())
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
