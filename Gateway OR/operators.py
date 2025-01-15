import random as ran
import numpy as np
from numba import njit
import hashlib

def parent_selection(evaluated_population):
    pop_size = len(evaluated_population[1])
    opponent_1, opponent_2 = ran.sample(range(pop_size), 2)
    o1_values = evaluated_population[1][opponent_1]
    o2_values = evaluated_population[1][opponent_2]
    if o1_values[0] > o2_values[0]:
        return opponent_1
    elif o1_values[0] < o2_values[0]:
        return opponent_2
    else:
        if o1_values[4] > o2_values[4]:
            return opponent_1
        elif o1_values[4] < o2_values[4]:
            return opponent_2
        else:
            return opponent_1 if o1_values[3] >= o2_values[3] else opponent_2

def crossover_per_process(crossover_probability, max_perc_of_num_tasks_for_crossover, cromossome_1, cromossome_2, reference_cromossome, cache_n_tokens, n_tokens_lock):
    if ran.random() < crossover_probability:
        max_length = len(cromossome_1) - 1
        max_number_of_tasks = 1 if max_perc_of_num_tasks_for_crossover == -1 else max(1, int(max_perc_of_num_tasks_for_crossover * max_length))
        number_of_tasks = ran.randint(1, max_number_of_tasks + 1)
        offspring_1, offspring_2 = np.copy(cromossome_1), np.copy(cromossome_2)

        def swap_task(i):
            count = 0
            while True:
                chosen_task = int(ran.random() * max_length)
                # if cromossome_1[chosen_task][-1] != -1 and cromossome_1[-1][chosen_task] != -1 and cromossome_2[chosen_task][-1] != -1 and cromossome_2[-1][chosen_task] != -1:
                #     break
                count += 1
                if count > len(cromossome_1):
                    return
            crossover_level = ran.random()
            if crossover_level < 1 / 3 or crossover_level > 2 / 3:
                offspring_1[chosen_task] = cromossome_2[chosen_task]
                offspring_2[chosen_task] = cromossome_1[chosen_task]
            if crossover_level >= 1 / 3:
                for j in range(len(offspring_1)):
                    offspring_1[j][chosen_task] = cromossome_2[j][chosen_task]
                    offspring_2[j][chosen_task] = cromossome_1[j][chosen_task]

        for i in range(number_of_tasks):
            swap_task(i)

        @njit
        def check_active_tasks(offspring):
            for row in range(len(offspring) - 2):
                if not is_there_at_least_one_row_active_task(offspring, row) and offspring[row][-1] != -1:
                    return False
                elif not is_there_at_least_two_row_active_tasks(offspring, row) and offspring[row][-1] in [1,2,3]:
                    offspring[row][-1] = 0
                elif is_there_at_least_two_row_active_tasks(offspring, row) and offspring[row][-1] == 0:
                    offspring[row][-1] = np.random.randint(1, 4)
            for col in range(1, len(offspring) - 1):
                if not is_there_at_least_one_column_active_task(offspring, col) and offspring[-1][col] != -1:
                    return False
                elif (not is_there_at_least_two_column_active_tasks(offspring, col) and offspring[-1][col] in [1,2,3]):
                    offspring[-1][col] = 0
                elif is_there_at_least_two_column_active_tasks(offspring, col) and offspring[-1][col] == 0:
                    offspring[-1][col] = np.random.randint(1, 4)
            return True

        ensure_connections(offspring_1, reference_cromossome)
        ensure_connections(offspring_2, reference_cromossome)
        checks = [check_active_tasks(offspring_1), check_active_tasks(offspring_2)]

        if all(checks):
            adjust_produced_consumed_tokens(offspring_1, cache_n_tokens, n_tokens_lock)
            adjust_produced_consumed_tokens(offspring_2, cache_n_tokens, n_tokens_lock)
            return offspring_2, offspring_1
    return cromossome_1, cromossome_2

def mutation(cromossome, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome, island, on_off_task_mutation_probability, max_perc_of_num_tasks_for_on_off_task_mutation, cache_n_tokens, n_tokens_lock):
    if ran.random() < task_mutation_probability:
        task_mutation(cromossome, max_perc_of_num_tasks_for_task_mutation, reference_cromossome, island, cache_n_tokens, n_tokens_lock)
    if ran.random() < gateway_mutation_probability:
        gateway_mutation(cromossome, max_perc_of_num_tasks_for_gateway_mutation, cache_n_tokens, n_tokens_lock)
    if ran.random() < on_off_task_mutation_probability:
        on_off_task_mutation(cromossome, reference_cromossome, max_perc_of_num_tasks_for_on_off_task_mutation, cache_n_tokens, n_tokens_lock)

@njit
def mutate_task(cromossome, chosen_task, reference_cromossome, reverse=False):
    cromossome_copy = cromossome[:]
    cromossome_len = len(cromossome)
    if reverse:
        if chosen_task == len(cromossome) - 2:
            return
        for j in range(cromossome_len):
            count = 0
            while True:
                new_output = int(ran.random() * (cromossome_len - 1))
                if cromossome[new_output][-1] != -1 and cromossome[-1][new_output] != -1:
                    break
                count += 1
                if count > len(cromossome):
                    cromossome = cromossome_copy[:]
                    return
            if reference_cromossome[chosen_task][new_output] == 1:
                break
        cromossome[chosen_task][new_output] ^= 1
        if cromossome[chosen_task][new_output] == 0:
            if not is_there_at_least_one_column_active_task(cromossome, new_output):
                possible_rows = np.array([i for i in range(cromossome_len - 2) if i != chosen_task])
                count = 0
                while True:
                    another_row = np.random.choice(possible_rows)
                    if cromossome[another_row][-1] != -1 and cromossome[-1][another_row] != -1:
                        break
                    count += 1
                    if count > len(cromossome):
                        cromossome = cromossome_copy[:]
                        return
                cromossome[another_row][new_output] = 1
                if is_there_at_least_two_row_active_tasks(cromossome, another_row) and cromossome[another_row][-1] == 0:
                    cromossome[another_row][-1] = np.random.randint(1, 4)
            if not is_there_at_least_one_row_active_task(cromossome, chosen_task):
                possible_columns = np.array([j for j in range(1, cromossome_len - 1) if j != new_output])
                count = 0
                while True:
                    another_col = np.random.choice(possible_columns)
                    if cromossome[another_col][-1] != -1 and cromossome[-1][another_col] != -1:
                        break
                    count += 1
                    if count > len(cromossome):
                        cromossome = cromossome_copy[:]
                        return
                cromossome[chosen_task][another_col] = 1
                if is_there_at_least_two_column_active_tasks(cromossome, another_col) and cromossome[-1][another_col] == 0:
                    cromossome[-1][another_col] = np.random.randint(1, 4)
        if not is_there_at_least_two_row_active_tasks(cromossome, chosen_task) and cromossome[chosen_task][-1] != 0:
            cromossome[chosen_task][-1] = 0
        elif is_there_at_least_two_row_active_tasks(cromossome, chosen_task) and cromossome[chosen_task][-1] == 0:
            cromossome[chosen_task][-1] = np.random.randint(1, 4)
        if (not is_there_at_least_two_column_active_tasks(cromossome, new_output) and cromossome[-1][new_output] != 0):
            cromossome[-1][new_output] = 0
        elif is_there_at_least_two_column_active_tasks(cromossome, new_output) and cromossome[-1][new_output] == 0:
            cromossome[-1][new_output] = np.random.randint(1, 4)
    else:
        if chosen_task == 0:
            return
        for j in range(cromossome_len):
            count = 0
            while True:
                new_input = int(ran.random() * (cromossome_len - 1))
                if cromossome[new_input][-1] != -1 and cromossome[-1][new_input] != -1:
                    break
                count += 1
                if count > len(cromossome):
                    cromossome = cromossome_copy[:]
                    return
            if reference_cromossome[new_input][chosen_task] == 1:
                break
        cromossome[new_input][chosen_task] ^= 1
        if cromossome[new_input][chosen_task] == 0:
            if not is_there_at_least_one_column_active_task(cromossome, chosen_task):
                possible_rows = np.array([i for i in range(cromossome_len - 2) if i != new_input])
                count = 0
                while True:
                    another_row = np.random.choice(possible_rows)
                    if cromossome[another_row][-1] != -1 and cromossome[-1][another_row] != -1:
                        break
                    count += 1
                    if count > len(cromossome):
                        cromossome = cromossome_copy[:]
                        return
                cromossome[another_row][chosen_task] = 1
                if is_there_at_least_two_row_active_tasks(cromossome, another_row) and cromossome[another_row][-1] == 0:
                    cromossome[another_row][-1] = np.random.randint(1, 4)
            if not is_there_at_least_one_row_active_task(cromossome, new_input):
                possible_columns = np.array([j for j in range(1, cromossome_len - 1) if j != chosen_task])
                count = 0
                while True:
                    another_col = np.random.choice(possible_columns)
                    if cromossome[another_col][-1] != -1 and cromossome[-1][another_col] != -1:
                        break
                    count += 1
                    if count > len(cromossome):
                        cromossome = cromossome_copy[:]
                        return
                cromossome[new_input][another_col] = 1
                if is_there_at_least_two_column_active_tasks(cromossome, another_col) and cromossome[-1][another_col] == 0:
                    cromossome[-1][another_col] = np.random.randint(1, 4)
        if not is_there_at_least_two_row_active_tasks(cromossome, new_input) and cromossome[new_input][-1] != 0:
            cromossome[new_input][-1] = 0
        elif is_there_at_least_two_row_active_tasks(cromossome, new_input) and cromossome[new_input][-1] == 0:
            cromossome[new_input][-1] = np.random.randint(1, 4)
        if (not is_there_at_least_two_column_active_tasks(cromossome, chosen_task) and cromossome[-1][chosen_task] != 0):
            cromossome[-1][chosen_task] = 0
        elif is_there_at_least_two_column_active_tasks(cromossome, chosen_task) and cromossome[-1][chosen_task] == 0:
            cromossome[-1][chosen_task] = np.random.randint(1, 4)

def adjust_tasks_and_gateways_for_task_mutation(cromossome, cromossome_len, chosen_task, new_input):
    if not is_there_at_least_one_column_active_task(cromossome, chosen_task):
        possible_rows = np.array([i for i in range(cromossome_len - 2) if i != new_input and cromossome[i][-1] != -1 and cromossome[-1][i] != -1])
        another_row = np.random.choice(possible_rows)
        cromossome[another_row][chosen_task] = 1
        if (not is_there_at_least_two_column_active_tasks(cromossome, another_row) and cromossome[-1][ another_row] != 0):
            cromossome[-1][another_row] = 0
        elif is_there_at_least_two_column_active_tasks(cromossome, another_row) and cromossome[-1][another_row] == 0:
            cromossome[-1][another_row] = np.random.randint(1, 4)
    if not is_there_at_least_one_row_active_task(cromossome, new_input):
        possible_columns = np.array([j for j in range(1, cromossome_len - 1) if j != chosen_task and cromossome[j][-1] != -1 and cromossome[-1][j] != -1])
        another_col = np.random.choice(possible_columns)
        cromossome[new_input][another_col] = 1
        if not is_there_at_least_two_row_active_tasks(cromossome, another_col) and cromossome[another_col][-1] != 0:
            cromossome[another_col][-1] = 0
        elif is_there_at_least_two_row_active_tasks(cromossome, another_col) and cromossome[another_col][-1] == 0:
            cromossome[another_col][-1] = np.random.randint(1, 4)

def adjust_produced_consumed_tokens(cromossome, cache_n_tokens, n_tokens_lock):
    cromossome, number_of_produced_single_tokens, number_of_consumed_single_tokens, number_of_produced_XOR_tokens, number_of_consumed_XOR_tokens, number_of_produced_AND_tokens, number_of_consumed_AND_tokens, number_of_produced_OR_tokens, number_of_consumed_OR_tokens = count_number_of_tokens(cromossome, cache_n_tokens, n_tokens_lock)
    if (number_of_produced_single_tokens + number_of_produced_XOR_tokens + number_of_produced_AND_tokens + number_of_produced_OR_tokens) > (number_of_consumed_single_tokens + number_of_consumed_XOR_tokens + number_of_consumed_AND_tokens + number_of_consumed_OR_tokens):
        increase_number_of_consumed_tokens(cromossome, number_of_produced_single_tokens, number_of_produced_XOR_tokens, number_of_produced_AND_tokens, number_of_produced_OR_tokens)
    elif (number_of_produced_single_tokens + number_of_produced_XOR_tokens + number_of_produced_AND_tokens + number_of_produced_OR_tokens) < (number_of_consumed_single_tokens + number_of_consumed_XOR_tokens + number_of_consumed_AND_tokens + number_of_consumed_OR_tokens):
        increase_number_of_produced_tokens(cromossome, number_of_consumed_single_tokens, number_of_consumed_XOR_tokens, number_of_consumed_AND_tokens, number_of_consumed_OR_tokens)

def hash_cromossomo_n_tokens(cromossomo):
    cromossomo_str = str(cromossomo.tolist())
    return hashlib.sha256(cromossomo_str.encode()).hexdigest()

def count_number_of_tokens(cromossome, cache_n_tokens, n_tokens_lock):
    # with n_tokens_lock:
    cromossomo_hash = hash_cromossomo_n_tokens(cromossome)
    cached_value = cache_n_tokens.get(cromossomo_hash)
    if isinstance(cached_value, tuple) and len(cached_value) == 8:
        if all(isinstance(count, int) for count in cached_value):
            return (cromossome,) + cached_value
    number_of_produced_single_tokens = count_produced_single_tokens(cromossome)
    number_of_consumed_single_tokens = count_consumed_single_tokens(cromossome)
    number_of_produced_XOR_tokens = count_produced_XOR_tokens(cromossome)
    number_of_consumed_XOR_tokens = count_consumed_XOR_tokens(cromossome)
    number_of_produced_AND_tokens = count_produced_AND_tokens(cromossome)
    number_of_consumed_AND_tokens = count_consumed_AND_tokens(cromossome)
    number_of_produced_OR_tokens = count_produced_OR_tokens(cromossome)
    number_of_consumed_OR_tokens = count_consumed_OR_tokens(cromossome)
    result = (number_of_produced_single_tokens, number_of_consumed_single_tokens, number_of_produced_XOR_tokens, number_of_consumed_XOR_tokens, number_of_produced_AND_tokens, number_of_consumed_AND_tokens, number_of_produced_OR_tokens, number_of_consumed_OR_tokens)
    if all(isinstance(count, int) for count in result):
        if cromossomo_hash not in cache_n_tokens:
            cache_n_tokens[cromossomo_hash] = result
    return (cromossome,) + result

def task_mutation(cromossome, max_perc_of_num_tasks_for_task_mutation, reference_cromossome, island, cache_n_tokens, n_tokens_lock):
    temp_cromossome = np.copy(cromossome)
    cromossome_len = len(cromossome)
    max_number_of_tasks = 1 if max_perc_of_num_tasks_for_task_mutation == -1 else max(1, int(max_perc_of_num_tasks_for_task_mutation * (cromossome_len - 1)))
    count1 = 0
    while True:
        unreachableOrNoProperEndTask = True
        number_of_tasks = ran.randint(1, max_number_of_tasks)
        for _ in range(number_of_tasks):
            count2 = 0
            while True:
                chosen_task = int(ran.random() * (cromossome_len - 1))
                if cromossome[chosen_task][-1] != -1 and cromossome[-1][chosen_task] != -1:
                    break
                count2 += 1
                if count2 > len(cromossome):
                    cromossome = temp_cromossome
                    return
            chosen_mutation_type = ran.random()
            mutate_task(cromossome, chosen_task, reference_cromossome, reverse=(chosen_mutation_type >= 1 / 2))
            adjust_produced_consumed_tokens(cromossome, cache_n_tokens, n_tokens_lock)
        for n in range(cromossome_len - 2):
            if not is_there_at_least_one_row_active_task(cromossome, n):
                unreachableOrNoProperEndTask = False
                break
        if unreachableOrNoProperEndTask:
            for n in range(1, cromossome_len - 1):
                if not is_there_at_least_one_column_active_task(cromossome, n):
                    unreachableOrNoProperEndTask = False
                    break
        if unreachableOrNoProperEndTask:
            return
        cromossome = temp_cromossome
        count1 += 1
        if count1 > len(cromossome):
            return

def on_off_task_mutation(cromossome, reference_cromossome, max_perc_of_num_tasks_for_on_off_task_mutation, cache_n_tokens, n_tokens_lock):
    tasks_on = np.array([i for i in range(1, len(cromossome) - 2) if cromossome[i][-1] != -1 and cromossome[-1][i] != -1])
    tasks_off = np.array([i for i in range(1, len(cromossome) - 2) if cromossome[i][-1] == -1 and cromossome[-1][i] == -1])
    if ran.random() < 0.5:
        max_num_to_activate = len(tasks_off)
        if max_num_to_activate > 0:
            num_to_activate = ran.randint(1, max(1, int(max_perc_of_num_tasks_for_on_off_task_mutation * max_num_to_activate)))
        else:
            return cromossome
        tasks_to_activate = np.random.choice(tasks_off, min(num_to_activate, len(tasks_off)), replace=False)
        for task in tasks_to_activate:
            cromossome = activate_task(cromossome, reference_cromossome, task, cache_n_tokens, n_tokens_lock)
    else:
        max_num_to_deactivate = len(tasks_on) - 1
        if max_num_to_deactivate > 0:
            num_to_deactivate = ran.randint(1, max(1, int(max_perc_of_num_tasks_for_on_off_task_mutation * max_num_to_deactivate)))
        else:
            return cromossome
        tasks_to_deactivate = np.random.choice(tasks_on, min(num_to_deactivate, len(tasks_on)), replace=False)
        for task in tasks_to_deactivate:
            cromossome = deactivate_task(cromossome, reference_cromossome, task, cache_n_tokens, n_tokens_lock)
    return cromossome

def activate_task(cromossome, reference_cromossome, task_index, cache_n_tokens, n_tokens_lock):
    cromossome[task_index][-1] = 0
    cromossome[-1][task_index] = 0
    create_connections(cromossome, reference_cromossome, task_index)
    update_gateway_states(cromossome, task_index)
    adjust_produced_consumed_tokens(cromossome, cache_n_tokens, n_tokens_lock)
    return cromossome

@njit
def create_connections(cromossome, reference_cromossome, task_index):
    possible_outputs = np.array([i for i in range(1, len(cromossome) - 1) if reference_cromossome[task_index][i] == 1])
    possible_inputs = np.array([i for i in range(len(cromossome) - 2) if reference_cromossome[i][task_index] == 1])
    if len(possible_outputs) > 0:
        chosen_output = np.random.choice(possible_outputs)
        cromossome[task_index][chosen_output] = 1
    for i in possible_outputs:
        if i != chosen_output and ran.random() < 0.5:
            cromossome[task_index][i] = 1
    if len(possible_inputs) > 0:
        chosen_input = np.random.choice(possible_inputs)
        cromossome[chosen_input][task_index] = 1
    for i in possible_inputs:
        if i != chosen_input and ran.random() < 0.5:
            cromossome[i][task_index] = 1

def deactivate_task(cromossome, reference_cromossome, task_index, cache_n_tokens, n_tokens_lock):
    cromossome[task_index, :] = 0
    cromossome[:, task_index] = 0
    cromossome[task_index][-1] = -1
    cromossome[-1][task_index] = -1
    ensure_connections(cromossome, reference_cromossome)
    update_gateway_states(cromossome, task_index)
    adjust_produced_consumed_tokens(cromossome, cache_n_tokens, n_tokens_lock)
    return cromossome

@njit
def ensure_connections(cromossome, reference_cromossome):
    cromossome_len = len(cromossome)
    for from_task in range(cromossome_len - 2):
        if not is_there_at_least_one_row_active_task(cromossome, from_task) and cromossome[from_task][-1] != -1:
            possible_tasks = [i for i in range(1, cromossome_len - 1) if i != from_task and reference_cromossome[from_task][i] == 1 and cromossome[i][-1] != -1 and cromossome[-1][i] != -1]
            if not possible_tasks:
                possible_tasks = [i for i in range(1, cromossome_len - 1) if i != from_task and cromossome[i][-1] != -1 and cromossome[-1][i] != -1]
            if possible_tasks:
                possible_tasks_array = np.array(possible_tasks)
                new_task = np.random.choice(possible_tasks_array)
                cromossome[from_task][new_task] = 1
    for to_task in range(1, cromossome_len - 1):
        if not is_there_at_least_one_column_active_task(cromossome, to_task) and cromossome[-1][to_task] != -1:
            possible_tasks = [i for i in range(cromossome_len - 2) if i != to_task and reference_cromossome[i][to_task] == 1 and cromossome[i][-1] != -1 and cromossome[-1][i] != -1]
            if not possible_tasks:
                possible_tasks = [i for i in range(cromossome_len - 2) if i != to_task and cromossome[i][-1] != -1 and cromossome[-1][i] != -1]
            if possible_tasks:
                possible_tasks_array = np.array(possible_tasks)
                new_task = np.random.choice(possible_tasks_array)
                cromossome[new_task][to_task] = 1

@njit
def update_gateway_states(cromossome, task_index):
    for row in range(len(cromossome) - 2):
        if not is_there_at_least_two_row_active_tasks(cromossome, row) and is_there_at_least_one_row_active_task(cromossome, row) and cromossome[row][-1] != 0:
            cromossome[row][-1] = 0
        elif is_there_at_least_two_row_active_tasks(cromossome, row) and cromossome[row][-1] not in [1,2,3]:
            cromossome[row][-1] = np.random.randint(1, 4)
    for col in range(1, len(cromossome) - 1):
        if not is_there_at_least_two_column_active_tasks(cromossome, col) and is_there_at_least_one_column_active_task(cromossome, col) and cromossome[-1][col] != 0:
            cromossome[-1][col] = 0
        elif is_there_at_least_two_column_active_tasks(cromossome, col) and cromossome[-1][col] not in [1,2,3]:
            cromossome[-1][col] = np.random.randint(1, 4)

def gateway_mutation(cromossome, max_perc_of_num_tasks_for_gateway_mutation, cache_n_tokens, n_tokens_lock):
    len_cromossome = len(cromossome)
    if max_perc_of_num_tasks_for_gateway_mutation == -1:
        number_of_tasks = 1
    else:
        max_number_of_tasks = max(1, int(max_perc_of_num_tasks_for_gateway_mutation * (len_cromossome - 1)))
        number_of_tasks = ran.randint(1, max_number_of_tasks)
    for _ in range(number_of_tasks):
        mutate_and_adjust(cromossome, len_cromossome, cache_n_tokens, n_tokens_lock)

def mutate_and_adjust(cromossome, len_cromossome, cache_n_tokens, n_tokens_lock):
    gateway_mutation_core(cromossome, len_cromossome)
    adjust_produced_consumed_tokens(cromossome, cache_n_tokens, n_tokens_lock)

@njit
def gateway_mutation_core(cromossome, len_cromossome):
    valid_rows = []
    valid_columns = []
    for i in range(len_cromossome - 2):
        if cromossome[i][-1] in [1, 2, 3]:
            valid_rows.append(i)
    for j in range(1, len_cromossome - 1):
        if cromossome[-1][j] in [1, 2, 3]:
            valid_columns.append(j)
    valid_rows = np.array(valid_rows)
    valid_columns = np.array(valid_columns)
    if len(valid_rows) == 0 and len(valid_columns) == 0:
        return
    chosen_operator_position = ran.random()
    if len(valid_rows) > 0 and len(valid_columns) > 0:
        if chosen_operator_position < 1 / 2:
            chosen_task = valid_rows[int(np.random.randint(len(valid_rows)))]
            element = cromossome[chosen_task][-1]
        else:
            chosen_task = valid_columns[int(np.random.randint(len(valid_columns)))]
            element = cromossome[-1][chosen_task]
    elif len(valid_rows) > 0:
        chosen_task = valid_rows[int(np.random.randint(len(valid_rows)))]
        element = cromossome[chosen_task][-1]
    else:
        chosen_task = valid_columns[int(np.random.randint(len(valid_columns)))]
        element = cromossome[-1][chosen_task]
    if element == 1:
        new_value = 2 if np.random.random() < 0.5 else 3
    elif element == 2:
        new_value = 1 if np.random.random() < 0.5 else 3
    elif element == 3:
        new_value = 1 if np.random.random() < 0.5 else 2
    if chosen_operator_position < 1 / 2:
        cromossome[chosen_task][-1] = new_value
    else:
        cromossome[-1][chosen_task] = new_value

@njit
def count_produced_single_tokens(cromossome):
    number_of_produced_single_tokens = 0
    for i in range(len(cromossome) - 2):
        if cromossome[i][-1] == 0:
            number_of_produced_single_tokens += 1
    return number_of_produced_single_tokens

@njit
def count_consumed_single_tokens(cromossome):
    number_of_consumed_single_tokens = 0
    for i in range(1, len(cromossome) - 1):
        if cromossome[-1][i] == 0:
            number_of_consumed_single_tokens += 1
    return number_of_consumed_single_tokens

@njit
def count_produced_XOR_tokens(cromossome):
    number_of_produced_XOR_tokens = 0
    for i in range(len(cromossome) - 2):
        if cromossome[i][-1] == 1:
            number_of_produced_XOR_tokens += 1
    return number_of_produced_XOR_tokens

@njit
def count_consumed_XOR_tokens(cromossome):
    number_of_consumed_XOR_tokens = 0
    for i in range(1, len(cromossome) - 1):
        if cromossome[-1][i] == 1:
            number_of_consumed_XOR_tokens += 1
    return number_of_consumed_XOR_tokens

@njit
def count_produced_AND_tokens(cromossome):
    number_of_produced_AND_tokens = 0
    for i in range(len(cromossome) - 2):
        if cromossome[i][-1] == 2:
            for j in range(1, len(cromossome[i]) - 1):
                number_of_produced_AND_tokens += cromossome[i][j]
    return number_of_produced_AND_tokens

@njit
def count_consumed_AND_tokens(cromossome):
    number_of_consumed_AND_tokens = 0
    for i in range(1, len(cromossome) - 1):
        if cromossome[-1][i] == 2:
            for j in range(len(cromossome[i]) - 2):
                number_of_consumed_AND_tokens += + cromossome[j][i]
    return number_of_consumed_AND_tokens

@njit
def count_produced_OR_tokens(cromossome):
    number_of_produced_OR_tokens = 0
    for i in range(len(cromossome) - 2):
        if cromossome[i][-1] == 3:
            for j in range(1, len(cromossome[i]) - 1):
                number_of_produced_OR_tokens += cromossome[i][j]
    return number_of_produced_OR_tokens

@njit
def count_consumed_OR_tokens(cromossome):
    number_of_consumed_OR_tokens = 0
    for i in range(1, len(cromossome) - 1):
        if cromossome[-1][i] == 3:
            for j in range(len(cromossome[i]) - 2):
                number_of_consumed_OR_tokens += + cromossome[j][i]
    return number_of_consumed_OR_tokens

@njit
def increase_number_of_produced_tokens(cromossome, number_of_consumed_single_tokens, number_of_consumed_XOR_tokens, number_of_consumed_AND_tokens, number_of_consumed_OR_tokens):
    n = len(cromossome)
    copy_of_output_gateways = np.array([task[-1] for task in cromossome])
    number_of_attempts = int(n * 0.5)
    for i in range(number_of_attempts):
        chosen_task = np.random.randint(0, n - 2)
        max_i = 0
        while (cromossome[chosen_task][-1] == 2 or cromossome[chosen_task][-1] == 3 or cromossome[chosen_task][-1] == -1) and max_i <= number_of_attempts:
            max_i += 1
            chosen_task = np.random.randint(0, n - 2)
        if max_i <= number_of_attempts:
            if cromossome[chosen_task][-1] == 1:
                if np.random.rand() < 0.5:
                    cromossome[chosen_task][-1] = 2
                else:
                    cromossome[chosen_task][-1] = 3
        number_of_produced_single_tokens = count_produced_single_tokens(cromossome)
        number_of_produced_XOR_tokens = count_produced_XOR_tokens(cromossome)
        number_of_produced_AND_tokens = count_produced_AND_tokens(cromossome)
        number_of_produced_OR_tokens = count_produced_OR_tokens(cromossome)
        max_i = 0
        while not ((number_of_produced_single_tokens + number_of_produced_XOR_tokens + number_of_produced_AND_tokens + number_of_produced_OR_tokens) == (number_of_consumed_single_tokens + number_of_consumed_XOR_tokens + number_of_consumed_AND_tokens + number_of_consumed_OR_tokens)) and max_i <= number_of_attempts:
            max_i += 1
            chosen_task = np.random.randint(0, n - 2)
            max_j = 0
            while (cromossome[chosen_task][-1] == 2 or cromossome[chosen_task][-1] == 3 or cromossome[chosen_task][-1] == -1) and max_j <= number_of_attempts:
                max_j += 1
                chosen_task = np.random.randint(0, n - 2)
            if max_j <= number_of_attempts:
                if cromossome[chosen_task][-1] == 1:
                    if np.random.rand() < 0.5:
                        cromossome[chosen_task][-1] = 2
                    else:
                        cromossome[chosen_task][-1] = 3
                    number_of_produced_single_tokens = count_produced_single_tokens(cromossome)
                    number_of_produced_XOR_tokens = count_produced_XOR_tokens(cromossome)
                    number_of_produced_AND_tokens = count_produced_AND_tokens(cromossome)
                    number_of_produced_OR_tokens = count_produced_OR_tokens(cromossome)
        if (number_of_produced_single_tokens + number_of_produced_XOR_tokens + number_of_produced_AND_tokens + number_of_produced_OR_tokens) != (number_of_consumed_single_tokens + number_of_consumed_XOR_tokens + number_of_consumed_AND_tokens + number_of_consumed_OR_tokens):
            cromossome[:, -1] = copy_of_output_gateways
    if i == number_of_attempts - 1:
        decrease_number_of_consumed_tokens(cromossome, count_produced_single_tokens(cromossome), count_produced_XOR_tokens(cromossome), count_produced_AND_tokens(cromossome), count_produced_OR_tokens(cromossome))

@njit
def increase_number_of_consumed_tokens(cromossome, number_of_produced_single_tokens, number_of_produced_XOR_tokens, number_of_produced_AND_tokens, number_of_produced_OR_tokens):
    n = len(cromossome)
    copy_of_input_gateways = cromossome[-1].copy()
    number_of_attempts = int(n * 0.5)
    for i in range(number_of_attempts):
        chosen_task = np.random.randint(1, n - 1)
        max_i = 0
        while (cromossome[-1][chosen_task] == 2 or cromossome[-1][chosen_task] == 3 or cromossome[-1][chosen_task] == -1) and max_i <= number_of_attempts:
            max_i += 1
            chosen_task = np.random.randint(1, n - 1)
        if max_i <= number_of_attempts:
            if cromossome[-1][chosen_task] == 1:
                if np.random.rand() < 0.5:
                    cromossome[-1][chosen_task] = 2
                else:
                    cromossome[-1][chosen_task] = 3
        number_of_consumed_single_tokens = count_consumed_single_tokens(cromossome)
        number_of_consumed_XOR_tokens = count_consumed_XOR_tokens(cromossome)
        number_of_consumed_AND_tokens = count_consumed_AND_tokens(cromossome)
        number_of_consumed_OR_tokens = count_consumed_OR_tokens(cromossome)
        max_i = 0
        while not ((number_of_produced_single_tokens + number_of_produced_XOR_tokens + number_of_produced_AND_tokens + number_of_produced_OR_tokens) == (number_of_consumed_single_tokens + number_of_consumed_XOR_tokens + number_of_consumed_AND_tokens + number_of_consumed_OR_tokens)) and max_i <= number_of_attempts:
            max_i += 1
            chosen_task = np.random.randint(1, n - 1)
            max_j = 0
            while (cromossome[-1][chosen_task] == 2 or cromossome[-1][chosen_task] == 3 or cromossome[-1][chosen_task] == -1) and max_j <= number_of_attempts:
                max_j += 1
                chosen_task = np.random.randint(1, n - 1)
            if max_j <= number_of_attempts:
                if cromossome[-1][chosen_task] == 1:
                    if np.random.rand() < 0.5:
                        cromossome[-1][chosen_task] = 2
                    else:
                        cromossome[-1][chosen_task] = 3
                    number_of_consumed_single_tokens = count_consumed_single_tokens(cromossome)
                    number_of_consumed_XOR_tokens = count_consumed_XOR_tokens(cromossome)
                    number_of_consumed_AND_tokens = count_consumed_AND_tokens(cromossome)
                    number_of_consumed_OR_tokens = count_consumed_OR_tokens(cromossome)
        if (number_of_produced_single_tokens + number_of_produced_XOR_tokens + number_of_produced_AND_tokens + number_of_produced_OR_tokens) != (number_of_consumed_single_tokens + number_of_consumed_XOR_tokens + number_of_consumed_AND_tokens + number_of_consumed_OR_tokens):
            cromossome[-1] = copy_of_input_gateways.copy()
    if i == number_of_attempts - 1:
        decrease_number_of_produced_tokens(cromossome, count_consumed_single_tokens(cromossome), count_consumed_XOR_tokens(cromossome), count_consumed_AND_tokens(cromossome), count_consumed_OR_tokens(cromossome))

@njit
def decrease_number_of_produced_tokens(cromossome, number_of_consumed_single_tokens, number_of_consumed_XOR_tokens, number_of_consumed_AND_tokens, number_of_consumed_OR_tokens):
    n = len(cromossome)
    copy_of_output_gateways = np.array([task[-1] for task in cromossome])
    number_of_attempts = int(n * 0.5)
    for i in range(number_of_attempts):
        chosen_task = np.random.randint(0, n - 2)
        max_i = 0
        while (cromossome[chosen_task][-1] == 0 or cromossome[chosen_task][-1] == 1 or cromossome[chosen_task][-1] == -1) and max_i <= number_of_attempts:
            max_i += 1
            chosen_task = np.random.randint(0, n - 2)
        if max_i <= number_of_attempts:
            cromossome[chosen_task][-1] = 1
        number_of_produced_single_tokens = count_produced_single_tokens(cromossome)
        number_of_produced_XOR_tokens = count_produced_XOR_tokens(cromossome)
        number_of_produced_AND_tokens = count_produced_AND_tokens(cromossome)
        number_of_produced_OR_tokens = count_produced_OR_tokens(cromossome)
        max_i = 0
        while not ((number_of_produced_single_tokens + number_of_produced_XOR_tokens + number_of_produced_AND_tokens + number_of_produced_OR_tokens) == (number_of_consumed_single_tokens + number_of_consumed_XOR_tokens + number_of_consumed_AND_tokens + number_of_consumed_OR_tokens)) and max_i <= number_of_attempts:
            max_i += 1
            chosen_task = np.random.randint(0, n - 2)
            max_j = 0
            while (cromossome[chosen_task][-1] == 0 or cromossome[chosen_task][-1] == 1 or cromossome[chosen_task][-1] == -1) and max_j <= number_of_attempts:
                max_j += 1
                chosen_task = np.random.randint(0, n - 2)
            if max_j <= number_of_attempts:
                cromossome[chosen_task][-1] = 1
                number_of_produced_single_tokens = count_produced_single_tokens(cromossome)
                number_of_produced_XOR_tokens = count_produced_XOR_tokens(cromossome)
                number_of_produced_AND_tokens = count_produced_AND_tokens(cromossome)
                number_of_produced_OR_tokens = count_produced_OR_tokens(cromossome)
        if (number_of_produced_single_tokens + number_of_produced_XOR_tokens + number_of_produced_AND_tokens + number_of_produced_OR_tokens) != (number_of_consumed_single_tokens + number_of_consumed_XOR_tokens + number_of_consumed_AND_tokens + number_of_consumed_OR_tokens):
            cromossome[:, -1] = copy_of_output_gateways

@njit
def decrease_number_of_consumed_tokens(cromossome, number_of_produced_single_tokens, number_of_produced_XOR_tokens, number_of_produced_AND_tokens, number_of_produced_OR_tokens):
    n = len(cromossome)
    copy_of_input_gateways = cromossome[-1].copy()
    number_of_attempts = int(n * 0.5)
    for i in range(number_of_attempts):
        chosen_task = np.random.randint(1, n - 1)
        max_i = 0
        while (cromossome[-1][chosen_task] == 0 or cromossome[-1][chosen_task] == 1 or cromossome[-1][chosen_task] == -1) and max_i <= number_of_attempts:
            max_i += 1
            chosen_task = np.random.randint(1, n - 1)
        if max_i <= number_of_attempts:
            cromossome[-1][chosen_task] = 1
        number_of_consumed_single_tokens = count_consumed_single_tokens(cromossome)
        number_of_consumed_XOR_tokens = count_consumed_XOR_tokens(cromossome)
        number_of_consumed_AND_tokens = count_consumed_AND_tokens(cromossome)
        number_of_consumed_OR_tokens = count_consumed_OR_tokens(cromossome)
        max_i = 0
        while not ((number_of_produced_single_tokens + number_of_produced_XOR_tokens + number_of_produced_AND_tokens + number_of_produced_OR_tokens) == (number_of_consumed_single_tokens + number_of_consumed_XOR_tokens + number_of_consumed_AND_tokens + number_of_consumed_OR_tokens)) and max_i <= number_of_attempts:
            max_i += 1
            chosen_task = np.random.randint(1, n - 1)
            max_j = 0
            while (cromossome[-1][chosen_task] == 0 or cromossome[-1][chosen_task] == 1 or cromossome[-1][chosen_task] == -1) and max_j <= number_of_attempts:
                max_j += 1
                chosen_task = np.random.randint(1, n - 1)
            if max_j <= number_of_attempts:
                cromossome[-1][chosen_task] = 1
                number_of_consumed_single_tokens = count_consumed_single_tokens(cromossome)
                number_of_consumed_XOR_tokens = count_consumed_XOR_tokens(cromossome)
                number_of_consumed_AND_tokens = count_consumed_AND_tokens(cromossome)
                number_of_consumed_OR_tokens = count_consumed_OR_tokens(cromossome)
        if (number_of_produced_single_tokens + number_of_produced_XOR_tokens + number_of_produced_AND_tokens + number_of_produced_OR_tokens) != (number_of_consumed_single_tokens + number_of_consumed_XOR_tokens + number_of_consumed_AND_tokens + number_of_consumed_OR_tokens):
            cromossome[-1] = copy_of_input_gateways.copy()

@njit
def is_there_at_least_one_row_active_task(cromossome, chosen_task):
    row = cromossome[chosen_task]
    for j in range(1, len(row) - 1):
        if row[j] == 1:
            return True
    return False

@njit
def is_there_at_least_one_column_active_task(cromossome, chosen_task):
    for j in range(len(cromossome) - 2):
        if cromossome[j][chosen_task] == 1:
            return True
    return False

@njit
def is_there_at_least_two_row_active_tasks(cromossome, chosen_task):
    count = 0
    row = cromossome[chosen_task]
    for j in range(1, len(row) - 1):
        if row[j] == 1:
            count += 1
            if count == 2:
                return True
    return False

@njit
def is_there_at_least_two_column_active_tasks(cromossome, chosen_task):
    count = 0
    for j in range(len(cromossome) - 2):
        if cromossome[j][chosen_task] == 1:
            count += 1
            if count == 2:
                return True
    return False

def elitism(population, elitism_percentage, sorted_evaluated_aux_population, sorted_evaluated_population, auxiliary_population, evaluated_aux_population, evaluated_population):
    elitism_count = round(len(population) * elitism_percentage)
    aux_pop_len = len(sorted_evaluated_aux_population)
    total_evaluation_aux_population = evaluated_aux_population[0]
    for i in range(elitism_count):
        aux_index = aux_pop_len - 1 - i
        aux_eval_value, aux_pop_idx = sorted_evaluated_aux_population[aux_index][0], int(
            sorted_evaluated_aux_population[aux_index][5])
        pop_eval_value, pop_idx = sorted_evaluated_population[i][0], int(sorted_evaluated_population[i][5])
        if aux_eval_value < pop_eval_value:
            auxiliary_population[aux_pop_idx] = population[pop_idx]
            total_evaluation_aux_population -= evaluated_aux_population[1][aux_pop_idx][0]
            aux_values = list(evaluated_aux_population[1][aux_pop_idx])
            aux_values[:5] = evaluated_population[1][pop_idx][:5]
            evaluated_aux_population[1][aux_pop_idx] = tuple(aux_values)
            total_evaluation_aux_population += evaluated_aux_population[1][aux_pop_idx][0]
        else:
            break
    evaluated_aux_population[0] = total_evaluation_aux_population

def count_active_inactive_tasks(cromossome):
    active_count = 0
    inactive_count = 0
    for i in range(1, len(cromossome) - 2):
        if cromossome[i][-1] != -1:
            active_count += 1
        else:
            inactive_count += 1
    return active_count, inactive_count

def check_condition(cromossome, source):
    for row in range(len(cromossome) - 2):
        if not is_there_at_least_one_row_active_task(cromossome, row) and cromossome[row][-1] != -1:
            print(1, cromossome, source)
            return False
        if not is_there_at_least_two_row_active_tasks(cromossome, row) and cromossome[row][-1] in [1,2,3]:
            print(2, cromossome, source)
            return False
        if is_there_at_least_two_row_active_tasks(cromossome, row) and cromossome[row][-1] == 0:
            print(3, cromossome, source)
            return False
    for col in range(1, len(cromossome) - 1):
        if not is_there_at_least_one_column_active_task(cromossome, col) and cromossome[-1][col] != -1:
            print(4, cromossome, source)
            return False
        if not is_there_at_least_two_column_active_tasks(cromossome, col) and cromossome[-1][col] in [1,2,3]:
            print(5, cromossome, source)
            return False
        if is_there_at_least_two_column_active_tasks(cromossome, col) and cromossome[-1][col] == 0:
            print(6, cromossome, source)
            return False
    return True