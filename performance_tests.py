from algorithm import Algorithm
from threading import Thread
from matplotlib import pyplot as plt
import cv2
import time


def read_image(filepath):
    return cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)


in_image_path = 'input_data/house_in.png'
ref_image_path = 'input_data/house_prewitt.png'
img_in = read_image(in_image_path)
img_ref = read_image(ref_image_path)


def create_model_params(alpha=1.0, beta=2.0, num_ants=512, evap_rate=0.05, decay=0.005, max_iter=50,
                        constr_steps=40, eps=0.01, init_pher=0.01):
    return {
        "alpha": alpha,
        "beta": beta,
        "number_of_ants": num_ants,
        "evaporation_rate": evap_rate,
        "pheromone_decay": decay,
        "max_iter": max_iter,
        "construction_steps": constr_steps,
        "epsilon": eps,
        "initial_pheromone": init_pher
    }


def test_alpha():
    mdl = create_model_params()
    alg = Algorithm(img_in, img_ref, mdl)
    target_fcn_arr = []
    alpha_arr = []

    for alpha in range(0, 51, 2):
        print(f'Alpha {alpha/10} is being tested')
        alg.model.alpha = alpha/10
        img_edge = alg.run_algorithm(is_verbose=False)
        tf_temp = alg.model.calculate_ssim(img_edge)
        target_fcn_arr.append(tf_temp)
        alpha_arr.append(alpha/10)
        print(f'End of loop for alpha={alpha/10}')

    plt.figure()
    plt.plot(alpha_arr, target_fcn_arr, 'r.-')
    plt.xlabel("Alpha")
    plt.ylabel("Target function")
    plt.savefig("performance_tests/alpha.png")


def test_beta():
    mdl = create_model_params()
    alg = Algorithm(img_in, img_ref, mdl)
    target_fcn_arr = []
    beta_arr = []

    for beta in range(0, 51, 2):
        print(f'Beta {beta/10} is being tested')
        alg.model.beta = beta/10
        img_edge = alg.run_algorithm(is_verbose=False)
        tf_temp = alg.model.calculate_ssim(img_edge)
        target_fcn_arr.append(tf_temp)
        beta_arr.append(beta/10)
        print(f'End of loop for beta={beta/10}')

    plt.figure()
    plt.plot(beta_arr, target_fcn_arr, 'r.-')
    plt.xlabel("Beta")
    plt.ylabel("Target function")
    plt.savefig("performance_tests/beta.png")


def test_number_of_ants():
    mdl = create_model_params()
    alg = Algorithm(img_in, img_ref, mdl)
    target_fcn_arr = []
    ant_num_arr = []

    for ant_num in range(200, 801, 20):
        print(f'Ant number {ant_num} is being tested')
        alg.model.number_of_ants = ant_num
        img_edge = alg.run_algorithm(is_verbose=False)
        tf_temp = alg.model.calculate_ssim(img_edge)
        target_fcn_arr.append(tf_temp)
        ant_num_arr.append(ant_num)
        print(f'End of loop for ant number={ant_num}')

    plt.figure()
    plt.plot(ant_num_arr, target_fcn_arr, 'r.-')
    plt.xlabel("Number of ants")
    plt.ylabel("Target function")
    plt.savefig("performance_tests/ant_num.png")


def test_evap_rate():
    mdl = create_model_params()
    alg = Algorithm(img_in, img_ref, mdl)
    target_fcn_arr = []
    evap_rate_arr = []

    for evap_rate in range(0, 101, 5):
        print(f'Evaporation rate {evap_rate/100} is being tested')
        alg.model.evaporation_rate = evap_rate/100
        img_edge = alg.run_algorithm(is_verbose=False)
        tf_temp = alg.model.calculate_ssim(img_edge)
        target_fcn_arr.append(tf_temp)
        evap_rate_arr.append(evap_rate/100)
        print(f'End of loop for evaporation rate={evap_rate/100}')

    plt.figure()
    plt.plot(evap_rate_arr, target_fcn_arr, 'r.-')
    plt.xlabel("Evaporation rate")
    plt.ylabel("Target function")
    plt.savefig("performance_tests/evap_rate.png")


def test_decay():
    mdl = create_model_params()
    alg = Algorithm(img_in, img_ref, mdl)
    target_fcn_arr = []
    decay_rate_arr = []

    for decay_rate in range(0, 501, 25):
        print(f'Pheromone decay rate {decay_rate/1000} is being tested')
        alg.model.pheromone_decay = decay_rate/1000
        img_edge = alg.run_algorithm(is_verbose=False)
        tf_temp = alg.model.calculate_ssim(img_edge)
        target_fcn_arr.append(tf_temp)
        decay_rate_arr.append(decay_rate/1000)
        print(f'End of loop for evaporation rate={decay_rate/1000}')

    plt.figure()
    plt.plot(decay_rate_arr, target_fcn_arr, 'r.-')
    plt.xlabel("Pheromone decay rate")
    plt.ylabel("Target function")
    plt.savefig("performance_tests/decay_rate.png")


def test_iter():
    mdl = create_model_params()
    alg = Algorithm(img_in, img_ref, mdl)
    target_fcn_arr = []
    iter_num_arr = []

    for iter_num in range(0, 301, 20):
        print(f'Iterations number {iter_num} is being tested')
        alg.model.max_iter = iter_num
        img_edge = alg.run_algorithm(is_verbose=False)
        tf_temp = alg.model.calculate_ssim(img_edge)
        target_fcn_arr.append(tf_temp)
        iter_num_arr.append(iter_num)
        print(f'End of loop for iterations num={iter_num}')

    plt.figure()
    plt.plot(iter_num_arr, target_fcn_arr, 'r.-')
    plt.xlabel("Number of iterations")
    plt.ylabel("Target function")
    plt.savefig("performance_tests/iter_num.png")


def test_const_steps():
    mdl = create_model_params()
    alg = Algorithm(img_in, img_ref, mdl)
    target_fcn_arr = []
    const_steps_arr = []

    for const_steps in range(0, 81, 5):
        print(f'Construction steps number {const_steps} is being tested')
        alg.model.construction_steps = const_steps
        img_edge = alg.run_algorithm(is_verbose=False)
        tf_temp = alg.model.calculate_ssim(img_edge)
        target_fcn_arr.append(tf_temp)
        const_steps_arr.append(const_steps)
        print(f'End of loop for construction steps={const_steps}')

    plt.figure()
    plt.plot(const_steps_arr, target_fcn_arr, 'r.-')
    plt.xlabel("Construction steps")
    plt.ylabel("Target function")
    plt.savefig("performance_tests/const_steps.png")


def test_init_pher():
    mdl = create_model_params()
    alg = Algorithm(img_in, img_ref, mdl)
    target_fcn_arr = []
    init_pher_arr = []

    for init_pher in range(0, 101, 5):
        print(f'Initial pheromone value {init_pher/100} is being tested')
        alg.model.initial_pheromone1 = init_pher/100
        img_edge = alg.run_algorithm(is_verbose=False)
        tf_temp = alg.model.calculate_ssim(img_edge)
        target_fcn_arr.append(tf_temp)
        init_pher_arr.append(init_pher/100)
        print(f'End of loop for initial pheromone={init_pher/100}')

    plt.figure()
    plt.plot(init_pher_arr, target_fcn_arr, 'r.-')
    plt.xlabel("Initial pheromone value")
    plt.ylabel("Target function")
    plt.savefig("performance_tests/init_pher.png")


thread_alpha = Thread(target=test_alpha)
thread_beta = Thread(target=test_beta)
thread_no_ants = Thread(target=test_number_of_ants)
thread_evap = Thread(target=test_evap_rate)
thread_decay = Thread(target=test_decay)
thread_iter = Thread(target=test_iter)
thread_const_steps = Thread(target=test_const_steps)
thread_init_pher = Thread(target=test_init_pher)

thread_list = {thread_alpha, thread_beta, thread_no_ants, thread_evap, thread_decay, thread_iter, thread_const_steps,
               thread_init_pher}


def main():
    for thread in thread_list:
        thread.start()

    for thread in thread_list:
        thread.join()

    print("End of testing")
    plt.show()


if __name__ == "__main__":
    start_time = time.clock()
    main()
    print(f'Time of program execution: {(time.clock() - start_time)/60} minutes')


