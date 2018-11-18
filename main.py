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
    # PARAM
    model_parameters = {
        "alpha": 1.0,
        "beta": 2.0,
        "number_of_ants": 512,
        "evaporation_rate": 0.05,
        "pheromone_decay": 0.005,
        "max_iter": 10,
        "construction_steps": 40,
        "epsilon": 0.01,
        "initial_pheromone": 0.1
    }
    # PARAM
    algorithm = Algorithm(img_in, img_ref, model_parameters)
    img_edge = algorithm.run_algorithm(True)
    print(f'Target fun: {algorithm.model.calculate_target_fcn(img_edge)}')
    cv2.imshow('Im edge', img_edge)
    cv2.imshow('Ref', img_ref)
    cv2.imshow('Pheromone', algorithm.pheromone_matrix / algorithm.pheromone_matrix.max())
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
