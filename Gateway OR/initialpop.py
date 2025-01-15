import fitness as fit
import petrinets as pn
import operators as op
import numpy as np
from numba import njit, prange
import hashlib

def process_trace(trace):
    return set(trace)

def create_alphabet(log):
    alphabet_set = set()
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
    log = [process_entry(entry) for entry in log]
    return log

def initialize_population(population_size, alphabet, log, xes_log, fitness_weight, precision_weight, generalization_weight, simplicity_weight, log_name, round, island, generation, number_of_islands, cache_fitness, cache_soundness, cache_petri_net, cache_n_tokens, fitness_lock, soundness_lock, petri_net_lock, n_tokens_lock):
    population = np.zeros((population_size, (len(alphabet) + 1), (len(alphabet)) + 1), dtype=np.int64)
    reference_cromossome = create_DFG(log, alphabet)
    reference_cromossome = np.array(reference_cromossome, dtype=np.int64)
    for i in range(population_size):
        population[i] = create_initial_individual(population[i], alphabet, reference_cromossome, cache_n_tokens, n_tokens_lock)
        while not pn.is_sound(population[i], alphabet, log_name, round, island, generation, cache_soundness, cache_petri_net, soundness_lock, petri_net_lock):
            population[i] = initialize_individual(len(alphabet) + 1)
            population[i] = create_initial_individual(population[i], alphabet, reference_cromossome, cache_n_tokens, n_tokens_lock)
    if island == 0:
        population[0] = reference_cromossome
    return (population, fit.evaluate_population(population, alphabet, xes_log, fitness_weight, precision_weight, generalization_weight, simplicity_weight, log_name, round, island, generation, cache_fitness, cache_petri_net, fitness_lock, petri_net_lock), reference_cromossome)

@njit
def initialize_individual(number_of_tasks):
    return np.zeros((number_of_tasks, number_of_tasks), dtype=np.int64)

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
    return DFG

def get_task_id(task, alphabet):
    return np.where(alphabet == task)[0][0]

def create_initial_individual(cromossome, alphabet, reference_cromossome, cache_n_tokens, n_tokens_lock):
    cromossome = create_tasks(alphabet, reference_cromossome, cromossome)
    cromossome = create_gateways(cromossome)
    cromossome, number_of_produced_single_tokens, number_of_consumed_single_tokens, number_of_produced_XOR_tokens, number_of_consumed_XOR_tokens, number_of_produced_AND_tokens, number_of_consumed_AND_tokens, number_of_produced_OR_tokens, number_of_consumed_OR_tokens = op.count_number_of_tokens(cromossome, cache_n_tokens, n_tokens_lock)
    cromossome = increase_or_decrease_tokens(cromossome, number_of_produced_single_tokens, number_of_consumed_single_tokens, number_of_produced_XOR_tokens, number_of_consumed_XOR_tokens, number_of_produced_AND_tokens, number_of_consumed_AND_tokens, number_of_produced_OR_tokens, number_of_consumed_OR_tokens)
    return cromossome

def create_tasks(alphabet, reference_cromossome, cromossome):
    total_of_tasks = len(alphabet) - 2
    alpha = 0.75  #quanto maior, aumenta a probabilidade de iniciar com um número maior de atividades off
    beta = 2.5  #quanto menor, aumenta a probabilidade de iniciar com um número maior de atividades off
    num_of_start_off_tasks = 0 + int((np.random.beta(alpha, beta)) * (total_of_tasks - 1))
    num_of_start_off_tasks = 0
    list_of_start_off_tasks = np.random.choice(range(1, total_of_tasks + 1), size=num_of_start_off_tasks, replace=False)
    for task in list_of_start_off_tasks:
        cromossome[task, -1] = -1
        cromossome[-1, task] = -1
    valid_start_tasks = [i for i in range(1, total_of_tasks + 1) if reference_cromossome[0][i] == 1 and i not in list_of_start_off_tasks]
    valid_start_tasks = np.array(valid_start_tasks)
    if valid_start_tasks.size == 0:
        valid_start_tasks = [i for i in range(1, total_of_tasks + 1) if i not in list_of_start_off_tasks]
    for _ in range(np.random.randint(1, int((total_of_tasks + len(list_of_start_off_tasks) + 3) * 0.5))):
        x1 = np.random.choice(valid_start_tasks)
        cromossome[0, x1] = 1
        count = 0
        while True:
            valid_next_tasks = [i for i in range(total_of_tasks + 2) if reference_cromossome[x1][i] == 1 and i not in list_of_start_off_tasks]
            valid_next_tasks = np.array(valid_next_tasks)
            if valid_next_tasks.size == 0:
                valid_next_tasks = [i for i in range(total_of_tasks + 2) if i not in list_of_start_off_tasks]
            x2 = np.random.choice(valid_next_tasks)
            cromossome[x1][x2] = 1
            x1 = x2
            if x2 == total_of_tasks + 1:
                break
            count += 1
            if count == total_of_tasks + 2:
                break
    for n in range(len(cromossome) - 2):
        if not op.is_there_at_least_one_row_active_task(cromossome, n) and cromossome[n, -1] != -1:
            valid_tasks = [i for i in range(1, len(cromossome) - 1) if reference_cromossome[n][i] == 1 and i not in list_of_start_off_tasks]
            valid_tasks = np.array(valid_tasks)
            if valid_tasks.size == 0:
                valid_tasks = [i for i in range(1, len(cromossome) - 1) if i not in list_of_start_off_tasks]
            cromossome[n][np.random.choice(valid_tasks)] = 1
    for n in range(1, len(cromossome) - 1):
        if not op.is_there_at_least_one_column_active_task(cromossome, n) and cromossome[-1, n] != -1:
            valid_tasks = [i for i in range(0, len(cromossome) - 2) if reference_cromossome[i][n] == 1 and i not in list_of_start_off_tasks]
            valid_tasks = np.array(valid_tasks)
            if valid_tasks.size == 0:
                valid_tasks = [i for i in range(0, len(cromossome) - 2) if i not in list_of_start_off_tasks]
            cromossome[np.random.choice(valid_tasks)][n] = 1
    return cromossome

@njit
def create_gateways(cromossome):
    for n in range(len(cromossome) - 2):
        if op.is_there_at_least_two_row_active_tasks(cromossome, n) and cromossome[n, -1] != -1:
            cromossome[n][-1] = 1
            # r = np.random.random()
            # if r <= 0.95:
            #     cromossome[n][-1] = 1
            # elif r > 0.98:
            #     cromossome[n][-1] = 2
            # else:
            #     cromossome[n][-1] = 3
    for n in range(1, len(cromossome) - 1):
        if op.is_there_at_least_two_column_active_tasks(cromossome, n) and cromossome[-1, n] != -1:
            cromossome[-1][n] = 1
            # r = np.random.random()
            # if r <= 0.95:
            #     cromossome[-1][n] = 1
            # elif r > 0.98:
            #     cromossome[-1][n] = 2
            # else:
            #     cromossome[-1][n] = 3
    return cromossome

def increase_or_decrease_tokens(cromossome, number_of_produced_single_tokens, number_of_consumed_single_tokens, number_of_produced_XOR_tokens, number_of_consumed_XOR_tokens, number_of_produced_AND_tokens, number_of_consumed_AND_tokens, number_of_produced_OR_tokens, number_of_consumed_OR_tokens):
    if (number_of_produced_single_tokens + number_of_produced_XOR_tokens + number_of_produced_AND_tokens + number_of_produced_OR_tokens) > (number_of_consumed_single_tokens + number_of_consumed_XOR_tokens + number_of_consumed_AND_tokens + number_of_consumed_OR_tokens):
        op.increase_number_of_consumed_tokens(cromossome, number_of_produced_single_tokens, number_of_produced_XOR_tokens, number_of_produced_AND_tokens, number_of_produced_OR_tokens)
    elif (number_of_produced_single_tokens + number_of_produced_XOR_tokens + number_of_produced_AND_tokens + number_of_produced_OR_tokens) < (number_of_consumed_single_tokens + number_of_consumed_XOR_tokens + number_of_consumed_AND_tokens + number_of_consumed_OR_tokens):
        op.increase_number_of_produced_tokens(cromossome, number_of_consumed_single_tokens, number_of_consumed_XOR_tokens, number_of_consumed_AND_tokens, number_of_consumed_OR_tokens)
    return cromossome