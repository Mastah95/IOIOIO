import model
import algorithm
from threading import Thread


def create_model_params(alpha, beta, num_ants, evap_rate, decay, max_iter, constr_steps, eps, init_pher):
    return\
    {
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
    print("a")
    # Run loop with different alpha, aggregate target_fcn results and plot it in the end


def test_beta():
    print("b")


def test_number_of_ants():
    print("c")


def test_evap_rate():
    print("d")


def test_decay():
    print("e")


def test_iter():
    print("f")


def test_const_steps():
    print("g")


def test_init_pher():
    print("h")


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


if __name__ == "__main__":
    main()


