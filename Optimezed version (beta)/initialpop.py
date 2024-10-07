import random as ran
import fitness as fit
import petrinets as pn
import operators as op
import numpy as np
from numba import njit, prange
from joblib import Parallel, delayed
from concurrent.futures import ThreadPoolExecutor

def process_trace(trace):
    return set(trace)

def create_alphabet(log):
    alphabet_set = set()

    # results = Parallel(n_jobs=-1)(delayed(process_trace)(trace) for trace in log)
    # for result in results:
    #     alphabet_set.update(result)
    for trace in log:
        result = process_trace(trace)
        alphabet_set.update(result)

    alphabet = np.array(sorted(alphabet_set))
    alphabet = np.insert(alphabet, 0, 'begin')
    alphabet = np.append(alphabet, 'end')
    print('Number of activities:', len(alphabet) - 2)
    return alphabet

def process_entry(entry):
    entry.insert(0, 'begin')
    entry.append('end')
    return entry

def process_log(log):
    # with ThreadPoolExecutor() as executor:
    #     log = list(executor.map(process_entry, log)
    log = [process_entry(entry) for entry in log]
    return log

def initialize_population(population_size, alphabet, log, xes_log, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight):
    population = np.zeros((population_size, (len(alphabet) + 1), (len(alphabet)) + 1), dtype=np.int64)
    reference_cromossome, average_enabled_tasks = create_DFG(log, alphabet)
    reference_cromossome = np.array(reference_cromossome, dtype=np.int64)  # Convertendo para NumPy array
    # for i in prange(population_size):
    for i in range(population_size):
        population[i] = create_initial_individual(population[i], alphabet, reference_cromossome)
        while not pn.is_sound(population[i], alphabet):
            population[i] = initialize_individual(len(alphabet) + 1)
            population[i] = create_initial_individual(population[i], alphabet, reference_cromossome)
    population[0] = reference_cromossome
    return (population, fit.evaluate_population(population, alphabet, xes_log, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight), reference_cromossome, average_enabled_tasks)

@njit
def initialize_individual(number_of_tasks):
    return np.zeros((number_of_tasks, number_of_tasks), dtype=np.int64)  # Explicitly specify dtype as np.int64

@njit
def create_empty_individual_task(number_of_tasks):
    return [0] * (number_of_tasks + 1)

def create_DFG(log, alphabet):
    DFG = [create_empty_individual_task(len(alphabet)) for _ in range(len(alphabet) + 1)]
    for trace in log:
        for k in range(len(trace) - 1):
            current_task = get_task_id(trace[k], alphabet)
            next_task = get_task_id(trace[k + 1], alphabet)
            DFG[current_task][next_task] = 1
    enabled_tasks = 0
    # for j in prange(len(DFG) - 1):
    for j in range(len(DFG) - 1):
        for k in range(1, len(DFG[j]) - 1):
            if DFG[j][k] == 1:
                enabled_tasks = enabled_tasks + 1
    return (DFG, enabled_tasks)


def get_task_id(task, alphabet):
    return np.where(alphabet == task)[0][0]


def create_initial_individual(cromossome, alphabet, reference_cromossome):
    cromossome = create_tasks(alphabet, reference_cromossome, cromossome)
    cromossome, number_of_produced_XOR_tokens, number_of_consumed_XOR_tokens, number_of_produced_AND_tokens, number_of_consumed_AND_tokens = create_AND_gateways(cromossome)
    cromossome = increase_or_decrease_tokens(cromossome, number_of_produced_XOR_tokens, number_of_consumed_XOR_tokens, number_of_produced_AND_tokens, number_of_consumed_AND_tokens)
    return cromossome

@njit
def create_tasks(alphabet, reference_cromossome, cromossome):
    valid_start_tasks = [i for i in range(1, len(alphabet) - 1) if reference_cromossome[0][i] == 1] ### new
    valid_start_tasks = np.array(valid_start_tasks)
    for n in range(np.random.randint(1, int(len(alphabet) * 0.5))):
        x1 = np.random.choice(valid_start_tasks)
        cromossome[0, x1] = 1
        while True:
            valid_next_tasks = [i for i in range(len(alphabet)) if reference_cromossome[x1][i] == 1]
            valid_next_tasks = np.array(valid_next_tasks)
            x2 = np.random.choice(valid_next_tasks)
            cromossome[x1][x2] = 1
            x1 = x2
            if x2 == len(alphabet) - 1:
                break
    # for n in prange(len(cromossome) - 2):
    for n in range(len(cromossome) - 2):
        if op.is_there_at_least_one_raw_active_task(cromossome, n) == False:
            valid_tasks = [i for i in range(1, len(cromossome) - 1) if reference_cromossome[n][i] == 1]
            valid_tasks = np.array(valid_tasks)
            if valid_tasks.size > 0:
                cromossome[n][np.random.choice(valid_tasks)] = 1
    # for n in prange(1, len(cromossome) - 1):
    for n in range(1, len(cromossome) - 1):
        if op.is_there_at_least_one_column_active_task(cromossome, n) == False:
            valid_tasks = [i for i in range(0, len(cromossome) - 2) if reference_cromossome[i][n] == 1]
            valid_tasks = np.array(valid_tasks)
            if valid_tasks.size > 0:
                cromossome[np.random.choice(valid_tasks)][n] = 1
    return cromossome

@njit
def process_cromossome_to_create_AND_gateways(cromossome):
    # for n in prange(len(cromossome) - 2):
    for n in range(len(cromossome) - 2):
        if np.random.random() < 0.5:
            cromossome[n][-1] = 1
    # for n in prange(1, len(cromossome) - 1):
    for n in range(1, len(cromossome) - 1):
        if np.random.random() < 0.5:
            cromossome[-1][n] = 1
    return cromossome

def create_AND_gateways(cromossome):
    cromossome = process_cromossome_to_create_AND_gateways(cromossome)

    # with ThreadPoolExecutor() as executor:
    #     future_produced_XOR = executor.submit(op.count_produced_XOR_tokens, cromossome)
    #     future_consumed_XOR = executor.submit(op.count_consumed_XOR_tokens, cromossome)
    #     future_produced_AND = executor.submit(op.count_produced_AND_tokens, cromossome)
    #     future_consumed_AND = executor.submit(op.count_consumed_AND_tokens, cromossome)
    #     number_of_produced_XOR_tokens = future_produced_XOR.result()
    #     number_of_consumed_XOR_tokens = future_consumed_XOR.result()
    #     number_of_produced_AND_tokens = future_produced_AND.result()
    #     number_of_consumed_AND_tokens = future_consumed_AND.result()
    number_of_produced_XOR_tokens = op.count_produced_XOR_tokens(cromossome)
    number_of_consumed_XOR_tokens = op.count_consumed_XOR_tokens(cromossome)
    number_of_produced_AND_tokens = op.count_produced_AND_tokens(cromossome)
    number_of_consumed_AND_tokens = op.count_consumed_AND_tokens(cromossome)

    return (cromossome, number_of_produced_XOR_tokens, number_of_consumed_XOR_tokens, number_of_produced_AND_tokens, number_of_consumed_AND_tokens)

def increase_or_decrease_tokens(cromossome, number_of_produced_XOR_tokens, number_of_consumed_XOR_tokens, number_of_produced_AND_tokens, number_of_consumed_AND_tokens):
    if (number_of_consumed_XOR_tokens != number_of_produced_XOR_tokens) or (number_of_consumed_AND_tokens != number_of_produced_AND_tokens):
        if np.random.rand() < 0.5:
            op.increase_number_of_produced_tokens(cromossome, number_of_consumed_XOR_tokens, number_of_consumed_AND_tokens)
        else:
            op.increase_number_of_consumed_tokens(cromossome, number_of_produced_XOR_tokens, number_of_produced_AND_tokens)
    return cromossome