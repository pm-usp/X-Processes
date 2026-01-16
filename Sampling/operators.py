import numpy as np
from numba import njit
import hashlib
from typing import Any, Dict, Optional, Sequence, Tuple, List, Union


@njit(cache=True)
def pick_two_distinct(pop_size: int):
    # assume pop_size >= 2
    a = np.random.randint(0, pop_size)
    b = np.random.randint(0, pop_size - 1)
    if b >= a:
        b += 1
    return a, b


@njit(cache=True)
def parent_selection(evaluated_population_idx1):
    pop_size = len(evaluated_population_idx1)
    opponent_1, opponent_2 = pick_two_distinct(pop_size)
    o1_values = evaluated_population_idx1[opponent_1]
    o2_values = evaluated_population_idx1[opponent_2]
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


@njit(cache=True)
def swap_task(max_length, cromossome_1, cromossome_2, offspring_1, offspring_2):
    chosen_task = int(np.random.random() * max_length)
    crossover_level = np.random.random()
    if cromossome_1[chosen_task][-1] != -1 and cromossome_1[-1][chosen_task] != -1 and cromossome_2[chosen_task][-1] != -1 and cromossome_2[-1][chosen_task] != -1:
        if crossover_level < 1 / 3 or crossover_level > 2 / 3:
            offspring_1[chosen_task] = cromossome_2[chosen_task]
            offspring_2[chosen_task] = cromossome_1[chosen_task]
        if crossover_level >= 1 / 3:
            for j in range(len(offspring_1)):
                offspring_1[j][chosen_task] = cromossome_2[j][chosen_task]
                offspring_2[j][chosen_task] = cromossome_1[j][chosen_task]
    else:
        offspring_1[chosen_task] = cromossome_2[chosen_task]
        offspring_2[chosen_task] = cromossome_1[chosen_task]
        for j in range(len(offspring_1)):
            offspring_1[j][chosen_task] = cromossome_2[j][chosen_task]
            offspring_2[j][chosen_task] = cromossome_1[j][chosen_task]


@njit(cache=True)
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


@njit(cache=True)
def _crossover_core(crossover_probability, max_perc_of_num_tasks_for_crossover, cromossome_1, cromossome_2, reference_cromossome):
    if np.random.random() >= crossover_probability:
        return False, cromossome_1, cromossome_2
    max_length = len(cromossome_1) - 1
    max_number_of_tasks = 1 if max_perc_of_num_tasks_for_crossover == -1 else max(1, int(max_perc_of_num_tasks_for_crossover * max_length))
    number_of_tasks = np.random.randint(1, max_number_of_tasks + 2)
    offspring_1 = np.copy(cromossome_1)
    offspring_2 = np.copy(cromossome_2)
    for _ in range(number_of_tasks):
        swap_task(max_length, cromossome_1, cromossome_2, offspring_1, offspring_2)
    fix_full_off_zombies(offspring_1)
    fix_full_off_zombies(offspring_2)
    fix_half_off_by_adjacency(offspring_1)
    fix_half_off_by_adjacency(offspring_2)
    ensure_connections(offspring_1, reference_cromossome)
    ensure_connections(offspring_2, reference_cromossome)
    if (not check_active_tasks(offspring_1)) or (not check_active_tasks(offspring_2)):
        return False, cromossome_1, cromossome_2
    fixed_post_1 = fix_full_off_zombies(offspring_1)
    fixed_post_2 = fix_full_off_zombies(offspring_2)
    if (fixed_post_1 + fixed_post_2) != 0:
        return False, cromossome_1, cromossome_2
    return True, offspring_1, offspring_2


#@njit(cache=True) - não aceita njit
def crossover_per_process(crossover_probability, max_perc_of_num_tasks_for_crossover, cromossome_1, cromossome_2, reference_cromossome, cache_n_tokens):
    ok, offspring_1, offspring_2 = _crossover_core(crossover_probability, max_perc_of_num_tasks_for_crossover, cromossome_1, cromossome_2, reference_cromossome)
    if not ok:
        return cromossome_1, cromossome_2
    adjust_produced_consumed_tokens(offspring_1, cache_n_tokens)
    adjust_produced_consumed_tokens(offspring_2, cache_n_tokens)
    if violates_selfloop_singleton(offspring_1) or violates_selfloop_singleton(offspring_2):
        return cromossome_1, cromossome_2
    return offspring_2, offspring_1


#@njit(cache=True) - não aceita njit
def mutation(cromossome, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome, on_off_task_mutation_probability, max_perc_of_num_tasks_for_on_off_task_mutation, cache_n_tokens, act_fix_mode):
    if np.random.random() < task_mutation_probability:
        task_mutation(cromossome, max_perc_of_num_tasks_for_task_mutation, reference_cromossome, cache_n_tokens)
    if np.random.random() < gateway_mutation_probability:
        gateway_mutation(cromossome, max_perc_of_num_tasks_for_gateway_mutation, cache_n_tokens)
    if np.random.random() < on_off_task_mutation_probability:
        if not act_fix_mode:
            on_off_task_mutation(cromossome, reference_cromossome, max_perc_of_num_tasks_for_on_off_task_mutation, cache_n_tokens)


@njit(cache=True)
def mutate_task(cromossome, chosen_task, reference_cromossome, reverse=False):
    cromossome_copy = np.copy(cromossome)
    cromossome_len = len(cromossome)
    if reverse:
        if chosen_task == len(cromossome) - 2:
            return
        for j in range(cromossome_len):
            count = 0
            while True:
                new_output = np.random.randint(1, cromossome_len - 1)
                if cromossome[new_output][-1] != -1 and cromossome[-1][new_output] != -1:
                    break
                count += 1
                if count > len(cromossome):
                    cromossome[:] = cromossome_copy
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
                    if cromossome[another_row][-1] != -1 and cromossome[-1][another_row] != -1 and (new_output != another_row or (cromossome[another_row][-1] != 0 and cromossome[-1][another_row] != 0)):
                        break
                    count += 1
                    if count > len(cromossome):
                        cromossome[:] = cromossome_copy
                        return
                cromossome[another_row][new_output] = 1
                if is_there_at_least_two_row_active_tasks(cromossome, another_row) and cromossome[another_row][-1] == 0:
                    cromossome[another_row][-1] = np.random.randint(1, 4)
            if not is_there_at_least_one_row_active_task(cromossome, chosen_task):
                possible_columns = np.array([j for j in range(1, cromossome_len - 1) if j != new_output])
                count = 0
                while True:
                    another_col = np.random.choice(possible_columns)
                    if cromossome[another_col][-1] != -1 and cromossome[-1][another_col] != -1 and (chosen_task != another_col or (cromossome[another_col][-1] != 0 and cromossome[-1][another_col] != 0)):
                        break
                    count += 1
                    if count > len(cromossome):
                        cromossome[:] = cromossome_copy
                        return
                cromossome[chosen_task][another_col] = 1
                if is_there_at_least_two_column_active_tasks(cromossome, another_col) and cromossome[-1][another_col] == 0:
                    cromossome[-1][another_col] = np.random.randint(1, 4)
        if not is_there_at_least_two_row_active_tasks(cromossome, chosen_task) and cromossome[chosen_task][-1] != 0:
            cromossome[chosen_task][-1] = 0
        elif is_there_at_least_two_row_active_tasks(cromossome, chosen_task) and cromossome[chosen_task][-1] == 0:
            cromossome[chosen_task][-1] = np.random.randint(1, 4)
        if (not is_there_at_least_two_column_active_tasks(cromossome, new_output) and cromossome[-1][new_output] != 0):
            if cromossome[new_output][new_output] != 0:
                cromossome[-1][new_output] = 0
            else:
                cromossome[:] = cromossome_copy
                return
        elif is_there_at_least_two_column_active_tasks(cromossome, new_output) and cromossome[-1][new_output] == 0:
            cromossome[-1][new_output] = np.random.randint(1, 4)
    else:
        if chosen_task == 0:
            return
        for j in range(cromossome_len):
            count = 0
            while True:
                new_input = np.random.randint(0, cromossome_len - 2)
                if cromossome[new_input][-1] != -1 and cromossome[-1][new_input] != -1:
                    break
                count += 1
                if count > len(cromossome):
                    cromossome[:] = cromossome_copy
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
                    if cromossome[another_row][-1] != -1 and cromossome[-1][another_row] != -1 and (chosen_task != another_row or (cromossome[another_row][-1] != 0 and cromossome[-1][another_row] != 0)):
                        break
                    count += 1
                    if count > len(cromossome):
                        cromossome[:] = cromossome_copy
                        return
                cromossome[another_row][chosen_task] = 1
                if is_there_at_least_two_row_active_tasks(cromossome, another_row) and cromossome[another_row][-1] == 0:
                    cromossome[another_row][-1] = np.random.randint(1, 4)
            if not is_there_at_least_one_row_active_task(cromossome, new_input):
                possible_columns = np.array([j for j in range(1, cromossome_len - 1) if j != chosen_task])
                count = 0
                while True:
                    another_col = np.random.choice(possible_columns)
                    if cromossome[another_col][-1] != -1 and cromossome[-1][another_col] != -1 and (new_input != another_col or (cromossome[another_col][-1] != 0 and cromossome[-1][another_col] != 0)):
                        break
                    count += 1
                    if count > len(cromossome):
                        cromossome[:] = cromossome_copy
                        return
                cromossome[new_input][another_col] = 1
                if is_there_at_least_two_column_active_tasks(cromossome, another_col) and cromossome[-1][another_col] == 0:
                    cromossome[-1][another_col] = np.random.randint(1, 4)
        if not is_there_at_least_two_row_active_tasks(cromossome, new_input) and cromossome[new_input][-1] != 0:
            cromossome[new_input][-1] = 0
        elif is_there_at_least_two_row_active_tasks(cromossome, new_input) and cromossome[new_input][-1] == 0:
            cromossome[new_input][-1] = np.random.randint(1, 4)
        if (not is_there_at_least_two_column_active_tasks(cromossome, chosen_task) and cromossome[-1][chosen_task] != 0):
            if cromossome[chosen_task][chosen_task] != 0:
                cromossome[-1][chosen_task] = 0
            else:
                cromossome[:] = cromossome_copy
                return
        elif is_there_at_least_two_column_active_tasks(cromossome, chosen_task) and cromossome[-1][chosen_task] == 0:
            cromossome[-1][chosen_task] = np.random.randint(1, 4)


@njit(cache=True)
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


#@njit(cache=True) - não aceita njit
def adjust_produced_consumed_tokens(cromossome, cache_n_tokens):
    cromossome, number_of_produced_single_tokens, number_of_consumed_single_tokens, number_of_produced_XOR_tokens, number_of_consumed_XOR_tokens, number_of_produced_AND_tokens, number_of_consumed_AND_tokens, number_of_produced_OR_tokens, number_of_consumed_OR_tokens = count_number_of_tokens(cromossome, cache_n_tokens)
    if (number_of_produced_single_tokens + number_of_produced_XOR_tokens + number_of_produced_AND_tokens + number_of_produced_OR_tokens) > (number_of_consumed_single_tokens + number_of_consumed_XOR_tokens + number_of_consumed_AND_tokens + number_of_consumed_OR_tokens):
        increase_number_of_consumed_tokens(cromossome, number_of_produced_single_tokens, number_of_produced_XOR_tokens, number_of_produced_AND_tokens, number_of_produced_OR_tokens)
    elif (number_of_produced_single_tokens + number_of_produced_XOR_tokens + number_of_produced_AND_tokens + number_of_produced_OR_tokens) < (number_of_consumed_single_tokens + number_of_consumed_XOR_tokens + number_of_consumed_AND_tokens + number_of_consumed_OR_tokens):
        increase_number_of_produced_tokens(cromossome, number_of_consumed_single_tokens, number_of_consumed_XOR_tokens, number_of_consumed_AND_tokens, number_of_consumed_OR_tokens)


#@njit(cache=True) - não aceita njit
def hash_cromossomo_n_tokens(cromossomo):
    arr = np.ascontiguousarray(np.asarray(cromossomo, dtype=np.int64))
    h = hashlib.sha256()
    h.update(arr.tobytes(order="C"))
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.dtype.str.encode("utf-8"))
    return h.hexdigest()


#@njit(cache=True) - não aceita njit
def count_number_of_tokens(cromossome, cache_n_tokens):
    cromossomo_hash = hash_cromossomo_n_tokens(cromossome)
    cached_value = cache_n_tokens.get(cromossomo_hash)
    if isinstance(cached_value, tuple) and len(cached_value) == 8:
        if all(isinstance(count, (int, np.integer)) for count in cached_value):
            return (cromossome,) + tuple(int(x) for x in cached_value)
    number_of_produced_single_tokens = count_produced_single_tokens(cromossome)
    number_of_consumed_single_tokens = count_consumed_single_tokens(cromossome)
    number_of_produced_XOR_tokens = count_produced_XOR_tokens(cromossome)
    number_of_consumed_XOR_tokens = count_consumed_XOR_tokens(cromossome)
    number_of_produced_AND_tokens = count_produced_AND_tokens(cromossome)
    number_of_consumed_AND_tokens = count_consumed_AND_tokens(cromossome)
    number_of_produced_OR_tokens = count_produced_OR_tokens(cromossome)
    number_of_consumed_OR_tokens = count_consumed_OR_tokens(cromossome)
    result = (int(number_of_produced_single_tokens), int(number_of_consumed_single_tokens), int(number_of_produced_XOR_tokens), int(number_of_consumed_XOR_tokens), int(number_of_produced_AND_tokens), int(number_of_consumed_AND_tokens), int(number_of_produced_OR_tokens), int(number_of_consumed_OR_tokens))
    if cromossomo_hash not in cache_n_tokens:
        cache_n_tokens[cromossomo_hash] = result
    return (cromossome,) + result


@njit(cache=True)
def _task_mutation_core(cromossome, max_perc_of_num_tasks_for_task_mutation, reference_cromossome):
    temp_cromossome = np.copy(cromossome)
    cromossome_len = len(cromossome)
    max_number_of_tasks = 1 if max_perc_of_num_tasks_for_task_mutation == -1 else max(1, int(max_perc_of_num_tasks_for_task_mutation * (cromossome_len - 1)))
    count1 = 0
    while True:
        unreachableOrNoProperEndTask = True
        number_of_tasks = np.random.randint(1, max_number_of_tasks + 1)
        for _ in range(number_of_tasks):
            count2 = 0
            while True:
                chosen_task = int(np.random.random() * (cromossome_len - 1))
                if cromossome[chosen_task, -1] != -1 and cromossome[-1, chosen_task] != -1:
                    break
                count2 += 1
                if count2 > cromossome_len:
                    cromossome[:] = temp_cromossome
                    return 0
            chosen_mutation_type = np.random.random()
            mutate_task(cromossome, chosen_task, reference_cromossome, reverse=(chosen_mutation_type >= 0.5))
            if violates_selfloop_singleton(cromossome):
                cromossome[:] = temp_cromossome
                return 0
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
            return 1
        cromossome[:] = temp_cromossome
        count1 += 1
        if count1 > cromossome_len:
            return 0

def task_mutation(cromossome, max_perc_of_num_tasks_for_task_mutation, reference_cromossome, cache_n_tokens):
    changed = _task_mutation_core(cromossome, max_perc_of_num_tasks_for_task_mutation, reference_cromossome)
    if changed == 1:
        adjust_produced_consumed_tokens(cromossome, cache_n_tokens)
    return


#@njit(cache=True) não aceita njit
def on_off_task_mutation(cromossome, reference_cromossome, max_perc_of_num_tasks_for_on_off_task_mutation, cache_n_tokens, max_attempts=10):
    tasks_on = np.array([i for i in range(1, len(cromossome) - 2) if cromossome[i][-1] != -1 and cromossome[-1][i] != -1])
    tasks_off = np.array([i for i in range(1, len(cromossome) - 2) if cromossome[i][-1] == -1 and cromossome[-1][i] == -1])
    if np.random.random() < 0.5:
        max_num_to_activate = len(tasks_off)
        if max_num_to_activate <= 0:
            return cromossome
        num_to_activate = np.random.randint(1, max(1, int(max_perc_of_num_tasks_for_on_off_task_mutation * max_num_to_activate)) + 1)
        candidates = tasks_off.copy()
        np.random.shuffle(candidates)
        candidates = candidates.tolist()
        activated = 0
        attempts = 0
        idx = 0
        no_progress = 0
        while activated < num_to_activate and attempts < max_attempts and len(candidates) > 0:
            task = candidates[idx]
            ok = activate_task(cromossome, reference_cromossome, task, cache_n_tokens)
            attempts += 1
            if ok:
                activated += 1
                candidates[idx] = candidates[-1]
                candidates.pop()
                no_progress = 0
                if idx >= len(candidates):
                    idx = 0
            else:
                idx += 1
                if idx >= len(candidates):
                    idx = 0
                no_progress += 1
                if no_progress >= len(candidates):
                    break
        return cromossome
    else:
        max_num_to_deactivate = len(tasks_on) - 1
        if max_num_to_deactivate > 0:
            num_to_deactivate = np.random.randint(1, max(1, int(max_perc_of_num_tasks_for_on_off_task_mutation * max_num_to_deactivate)) + 1)
        else:
            return cromossome
        tasks_to_deactivate = np.random.choice(tasks_on, min(num_to_deactivate, len(tasks_on)), replace=False)
        for task in tasks_to_deactivate:
            deactivate_task(cromossome, reference_cromossome, task, cache_n_tokens)
        return cromossome


@njit(cache=True)
def _activate_task_core(cromossome, reference_cromossome, task_index):
    cromossome_copy = np.copy(cromossome)
    cromossome[task_index, :] = 0
    cromossome[:, task_index] = 0
    cromossome[task_index, -1] = 0
    cromossome[-1, task_index] = 0
    if not create_connections(cromossome, reference_cromossome, task_index):
        cromossome[:] = cromossome_copy
        return False
    update_gateway_states(cromossome)
    return True

def activate_task(cromossome, reference_cromossome, task_index, cache_n_tokens):
    ok = _activate_task_core(cromossome, reference_cromossome, task_index)
    if ok:
        adjust_produced_consumed_tokens(cromossome, cache_n_tokens)
    return ok

@njit(cache=True)
def _deactivate_task_core_stage1(cromossome, reference_cromossome, task_index):
    cromossome[task_index, :] = 0
    cromossome[:, task_index] = 0
    cromossome[task_index, -1] = -1
    cromossome[-1, task_index] = -1
    ensure_connections(cromossome, reference_cromossome)
    update_gateway_states(cromossome)

def deactivate_task(cromossome, reference_cromossome, task_index, cache_n_tokens):
    cromossome_copy = np.copy(cromossome)
    _deactivate_task_core_stage1(cromossome, reference_cromossome, task_index)
    adjust_produced_consumed_tokens(cromossome, cache_n_tokens)
    if violates_selfloop_singleton(cromossome):
        cromossome[:] = cromossome_copy
        return False
    return True


@njit(cache=True)
def create_connections(cromossome, reference_cromossome, task_index):
    def is_off_complete(i: int) -> bool:
        return cromossome[i, -1] == -1 and cromossome[-1, i] == -1
    possible_outputs = np.array([i for i in range(1, len(cromossome) - 1) if i != task_index and reference_cromossome[task_index][i] == 1 and not is_off_complete(i)])
    possible_inputs = np.array([i for i in range(len(cromossome) - 2) if i != task_index and reference_cromossome[i][task_index] == 1 and not is_off_complete(i)])
    if len(possible_outputs) == 0 or len(possible_inputs) == 0:
        return False
    chosen_output = np.random.choice(possible_outputs)
    cromossome[task_index][chosen_output] = 1
    chosen_input = np.random.choice(possible_inputs)
    cromossome[chosen_input][task_index] = 1
    for i in possible_outputs:
        if i != chosen_output and np.random.random() < 0.5:
            cromossome[task_index][i] = 1
    for i in possible_inputs:
        if i != chosen_input and np.random.random() < 0.5:
            cromossome[i][task_index] = 1
    if reference_cromossome[task_index][task_index] == 1 and np.random.random() < 0.5:
        cromossome[task_index][task_index] = 1
    return True


@njit(cache=True)
def violates_selfloop_singleton(cromossome) -> bool:
    m = len(cromossome)
    last = m - 1
    end = m - 2
    for i in range(0, end):
        if cromossome[i, last] != 0:
            continue
        if cromossome[i, i] != 1:
            continue
        out_cnt = 0
        for j in range(1, end + 1):
            if cromossome[i, j] == 1:
                out_cnt += 1
                if out_cnt > 1:
                    break
        if out_cnt == 1:
            return True
    for j in range(1, end + 1):
        if cromossome[last, j] != 0:
            continue
        if cromossome[j, j] != 1:
            continue
        in_cnt = 0
        for i in range(0, end):
            if cromossome[i, j] == 1:
                in_cnt += 1
                if in_cnt > 1:
                    break
        if in_cnt == 1:
            return True
    return False


@njit(cache=True)
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


@njit(cache=True)
def update_gateway_states(cromossome):
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


@njit(cache=True)
def gateway_mutation_core_pre(cromossome, max_perc_of_num_tasks_for_gateway_mutation):
    len_cromossome = len(cromossome)
    if max_perc_of_num_tasks_for_gateway_mutation == -1:
        number_of_tasks = 1
    else:
        max_number_of_tasks = max(1, int(max_perc_of_num_tasks_for_gateway_mutation * (len_cromossome - 1)))
        number_of_tasks = np.random.randint(1, max_number_of_tasks + 1)
    return number_of_tasks, len_cromossome


def gateway_mutation(cromossome, max_perc_of_num_tasks_for_gateway_mutation, cache_n_tokens):
    number_of_tasks, cromossome_size = gateway_mutation_core_pre(cromossome, max_perc_of_num_tasks_for_gateway_mutation)
    for _ in range(number_of_tasks):
        gateway_mutation_core_post(cromossome, cromossome_size)
        adjust_produced_consumed_tokens(cromossome, cache_n_tokens)


@njit(cache=True)
def gateway_mutation_core_post(cromossome, len_cromossome):
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
    chosen_operator_position = np.random.random()
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


@njit(cache=True)
def count_produced_single_tokens(cromossome):
    number_of_produced_single_tokens = 0
    for i in range(len(cromossome) - 2):
        if cromossome[i][-1] == 0:
            number_of_produced_single_tokens += 1
    return number_of_produced_single_tokens


@njit(cache=True)
def count_consumed_single_tokens(cromossome):
    number_of_consumed_single_tokens = 0
    for i in range(1, len(cromossome) - 1):
        if cromossome[-1][i] == 0:
            number_of_consumed_single_tokens += 1
    return number_of_consumed_single_tokens


@njit(cache=True)
def count_produced_XOR_tokens(cromossome):
    number_of_produced_XOR_tokens = 0
    for i in range(len(cromossome) - 2):
        if cromossome[i][-1] == 1:
            number_of_produced_XOR_tokens += 1
    return number_of_produced_XOR_tokens


@njit(cache=True)
def count_consumed_XOR_tokens(cromossome):
    number_of_consumed_XOR_tokens = 0
    for i in range(1, len(cromossome) - 1):
        if cromossome[-1][i] == 1:
            number_of_consumed_XOR_tokens += 1
    return number_of_consumed_XOR_tokens


@njit(cache=True)
def count_produced_AND_tokens(cromossome):
    number_of_produced_AND_tokens = 0
    for i in range(len(cromossome) - 2):
        if cromossome[i][-1] == 2:
            for j in range(1, len(cromossome[i]) - 1):
                number_of_produced_AND_tokens += cromossome[i][j]
    return number_of_produced_AND_tokens


@njit(cache=True)
def count_consumed_AND_tokens(cromossome):
    number_of_consumed_AND_tokens = 0
    for i in range(1, len(cromossome) - 1):
        if cromossome[-1][i] == 2:
            for j in range(len(cromossome[i]) - 2):
                number_of_consumed_AND_tokens += + cromossome[j][i]
    return number_of_consumed_AND_tokens


@njit(cache=True)
def count_produced_OR_tokens(cromossome):
    number_of_produced_OR_tokens = 0
    for i in range(len(cromossome) - 2):
        if cromossome[i][-1] == 3:
            for j in range(1, len(cromossome[i]) - 1):
                number_of_produced_OR_tokens += cromossome[i][j]
    return number_of_produced_OR_tokens


@njit(cache=True)
def count_consumed_OR_tokens(cromossome):
    number_of_consumed_OR_tokens = 0
    for i in range(1, len(cromossome) - 1):
        if cromossome[-1][i] == 3:
            for j in range(len(cromossome[i]) - 2):
                number_of_consumed_OR_tokens += + cromossome[j][i]
    return number_of_consumed_OR_tokens


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
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


@njit(cache=True)
def is_there_at_least_one_row_active_task(cromossome, chosen_task):
    row = cromossome[chosen_task]
    for j in range(1, len(row) - 1):
        if row[j] == 1:
            return True
    return False


@njit(cache=True)
def is_there_at_least_one_column_active_task(cromossome, chosen_task):
    for j in range(len(cromossome) - 2):
        if cromossome[j][chosen_task] == 1:
            return True
    return False


@njit(cache=True)
def is_there_at_least_two_row_active_tasks(cromossome, chosen_task):
    count = 0
    row = cromossome[chosen_task]
    for j in range(1, len(row) - 1):
        if row[j] == 1:
            count += 1
            if count == 2:
                return True
    return False


@njit(cache=True)
def is_there_at_least_two_column_active_tasks(cromossome, chosen_task):
    count = 0
    for j in range(len(cromossome) - 2):
        if cromossome[j][chosen_task] == 1:
            count += 1
            if count == 2:
                return True
    return False


#@njit(cache=True)
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


@njit(cache=True)
def count_active_inactive_tasks(cromossome):
    active_count = 0
    inactive_count = 0
    for i in range(1, len(cromossome) - 2):
        if cromossome[i][-1] != -1:
            active_count += 1
        else:
            inactive_count += 1
    return active_count, inactive_count


@njit(cache=True)
def fix_full_off_zombies(cromossome):
    n = cromossome.shape[0]
    last = n - 1
    fixed = 0
    for i in range(0, n - 2):
        if cromossome[i, last] == -1 and cromossome[last, i] == -1:
            dirty = False
            for j in range(0, n - 1):
                if cromossome[i, j] != 0:
                    dirty = True
                    break
            if not dirty:
                for j in range(0, n - 1):
                    if cromossome[j, i] != 0:
                        dirty = True
                        break
            if dirty:
                fixed += 1
                for j in range(0, n - 1):
                    cromossome[i, j] = 0
                for j in range(0, n - 1):
                    cromossome[j, i] = 0
                cromossome[i, last] = -1
                cromossome[last, i] = -1
    return fixed


@njit(cache=True)
def fix_half_off_by_adjacency(cromossome):
    """
    Corrige casos "meia OFF" (exatamente um metadado é -1) com base em adjacência.

    Definições:
      R = existe pelo menos um 1 na linha i (saídas), em colunas 1..n-2
      C = existe pelo menos um 1 na coluna i (entradas), em linhas 0..n-3

    Política:
      - Se R=1 e C=1: força ON, troca -1 por 0 no(s) metadado(s) -1
      - Se R=0 e C=0: força OFF completa, seta ambos metadados -1 e zera linha e coluna
      - Se R xor C: força ON, troca -1 por 0 no(s) metadado(s) -1

    Retorna:
      (n_half_fixed, n_forced_on, n_forced_off)
    """
    n = cromossome.shape[0]
    last = n - 1
    n_half_fixed = 0
    n_forced_on = 0
    n_forced_off = 0
    for i in range(0, n - 1):
        split_meta = cromossome[i, last]
        join_meta = cromossome[last, i]
        if (split_meta == -1) != (join_meta == -1):
            n_half_fixed += 1
            row_has = False
            for j in range(1, n - 1):
                if cromossome[i, j] == 1:
                    row_has = True
                    break
            col_has = False
            for j in range(0, n - 2):
                if cromossome[j, i] == 1:
                    col_has = True
                    break
            if (not row_has) and (not col_has):
                for j in range(0, n - 1):
                    cromossome[i, j] = 0
                for j in range(0, n - 1):
                    cromossome[j, i] = 0
                cromossome[i, last] = -1
                cromossome[last, i] = -1
                n_forced_off += 1
            else:
                if split_meta == -1:
                    cromossome[i, last] = 0
                if join_meta == -1:
                    cromossome[last, i] = 0
                n_forced_on += 1
    return n_half_fixed, n_forced_on, n_forced_off


@njit(cache=True)
def count_half_off(cromossome):
    n = cromossome.shape[0]
    last = n - 1
    cnt = 0
    for i in range(1, n - 2):
        if (cromossome[i, last] == -1) != (cromossome[last, i] == -1):
            cnt += 1
    return cnt


def sanity_check_cromossome(
    cromossome: Any,
    *,
    alphabet: Optional[Sequence[str]] = None,
    context: Optional[Union[str, Dict[str, Any]]] = None,
    max_positions: int = 50,
    print_full: bool = True,
) -> bool:
    """
    Checa regras de sanidade do cromossomo (matriz m x m) sem modificar nada.
    Imprime apenas quando encontrar violações. Não levanta exceção.
    Retorna True se OK, False se encontrou violação (ou se a checagem falhou).

    Notas:
      - Rejeita dtype unsigned e bool, pois o encoding usa -1, 2, 3.
      - Metadados são sempre validados via _strict_int (não converte strings nem objetos).
      - Para evitar explosão de memória, armazena apenas amostras de coordenadas (até max_positions),
        mas mantém contagens totais.
      - Regras 6.1 e 6.2 são SEMPRE aplicadas (exatamente como você definiu).
    """
    issues: List[Tuple[str, str]] = []

    def _add(rule: str, msg: str) -> None:
        issues.append((rule, msg))

    def _ctx_str() -> str:
        if context is None:
            return ""
        if isinstance(context, str):
            return context
        try:
            return " ".join(f"{k}={context[k]}" for k in sorted(context))
        except Exception:
            return str(context)

    def _strict_int(v: Any) -> Tuple[Optional[int], bool, str]:
        """
        Aceita apenas:
          - int/np.integer (exceto bool)
          - float/np.floating finito e exatamente inteiro (ex.: 1.0)
        Rejeita strings, None, NaN, inf, float não inteiro, bool.
        """
        try:
            if isinstance(v, (bool, np.bool_)):
                return None, False, "bool inválido"
            if isinstance(v, (int, np.integer)):
                return int(v), True, ""
            if isinstance(v, (float, np.floating)):
                if not np.isfinite(v):
                    return None, False, "float não finito"
                iv = int(v)
                if float(iv) != float(v):
                    return None, False, "float não inteiro"
                return iv, True, ""
            return None, False, f"tipo inválido ({type(v)})"
        except Exception as e:
            return None, False, f"falha ao interpretar ({type(e).__name__}: {e})"

    def _axis_label(idx: int, *, axis: str, end_idx: int, last: int) -> str:
        # axis: "row" ou "col"
        if axis == "row" and idx == last:
            return "meta_join"
        if axis == "col" and idx == last:
            return "meta_split"
        if idx == 0:
            return "begin"
        if idx == end_idx:
            return "end"
        if alphabet is not None:
            try:
                if 0 <= idx < len(alphabet):
                    return str(alphabet[idx])
            except Exception:
                pass
        return str(idx)

    def _coords_as_str(
        arr: np.ndarray,
        coords: List[Tuple[int, int]],
        *,
        end_idx: int,
        last: int,
    ) -> str:
        if not coords:
            return "[]"
        shown = coords[:max_positions]
        parts: List[str] = []
        for r, c_ in shown:
            r = int(r)
            c_ = int(c_)
            try:
                val = arr[r, c_]
            except Exception:
                val = None
            parts.append(
                f"({r},{c_}) [{_axis_label(r, axis='row', end_idx=end_idx, last=last)}"
                f"->{_axis_label(c_, axis='col', end_idx=end_idx, last=last)}]={val!r}"
            )
        if len(coords) > max_positions:
            parts.append(f"(+{len(coords) - max_positions} restantes; aumente max_positions)")
        return ", ".join(parts)

    def _print_report(arr: Any) -> None:
        if not issues:
            return
        c_arr = np.asarray(arr)
        hdr = "[sanity] VIOLAÇÕES DETECTADAS"
        ctx = _ctx_str()
        if ctx:
            hdr += f" | {ctx}"
        print(hdr)
        print(f"shape={getattr(c_arr, 'shape', None)} dtype={getattr(c_arr, 'dtype', None)}")
        for rule, msg in issues:
            print(f"  R{rule}: {msg}")
        if print_full:
            try:
                with np.printoptions(linewidth=240, threshold=c_arr.size):
                    print("cromossome=")
                    print(c_arr)
            except Exception:
                print("cromossome=")
                print(repr(arr))

    def _sample_indices_1d(cond: np.ndarray, limit: int) -> List[int]:
        out: List[int] = []
        for i in range(int(cond.shape[0])):
            if bool(cond[i]):
                out.append(int(i))
                if len(out) >= limit:
                    break
        return out

    def _sample_coords_from_mask(mask: np.ndarray, limit: int) -> List[Tuple[int, int]]:
        coords: List[Tuple[int, int]] = []
        rows = int(mask.shape[0])
        for i in range(rows):
            if not bool(mask[i].any()):
                continue
            cols = np.flatnonzero(mask[i])
            if cols.size:
                remaining = limit - len(coords)
                for c_ in cols[:remaining]:
                    coords.append((int(i), int(c_)))
                    if len(coords) >= limit:
                        return coords
        return coords

    try:
        c = np.asarray(cromossome)

        # Shape
        if c.ndim != 2 or c.shape[0] != c.shape[1]:
            _add("shape", f"cromossome não é matriz quadrada 2D, shape={getattr(c, 'shape', None)}")
            _print_report(c)
            return False

        m = int(c.shape[0])
        if m < 3:
            _add("shape", f"m={m} muito pequeno para este encoding")
            _print_report(c)
            return False

        last = m - 1
        end_idx = m - 2

        # dtype guardrail
        if isinstance(c, np.ndarray) and c.dtype.kind in ("u", "b"):
            _add(
                "dtype",
                f"dtype={c.dtype} incompatível com encoding (precisa representar -1, 2, 3; unsigned/bool não serve)",
            )
            _print_report(c)
            return False

        A = c[:-1, :-1]  # (m-1) x (m-1)
        is_int_like = isinstance(c, np.ndarray) and c.dtype.kind == "i"

        # 1) crom[:,0] == 0
        if is_int_like:
            cond = (c[:, 0] != 0)
            total = int(np.count_nonzero(cond))
            if total:
                idxs = _sample_indices_1d(cond, max_positions)
                coords = [(int(r), 0) for r in idxs]
                _add(
                    "1",
                    f"crom[:,0] deveria ser todo 0, mas há {total} não-zero. exemplos: "
                    f"{_coords_as_str(c, coords, end_idx=end_idx, last=last)}",
                )
        else:
            bad: List[Tuple[int, Any, str]] = []
            total = 0
            for i in range(m):
                v = c[i, 0]
                iv, ok, why = _strict_int(v)
                if (not ok) or (iv != 0):
                    total += 1
                    if len(bad) < max_positions:
                        bad.append((i, v, why))
            if total:
                _add("1", f"crom[:,0] deveria ser todo 0, mas há {total} inválidos. exemplos(i,valor,motivo)={bad}")

        # 2) crom[-2,:] == 0
        if is_int_like:
            cond = (c[end_idx, :] != 0)
            total = int(np.count_nonzero(cond))
            if total:
                idxs = _sample_indices_1d(cond, max_positions)
                coords = [(end_idx, int(col)) for col in idxs]
                _add(
                    "2",
                    f"crom[-2,:] (linha do end, idx={end_idx}) deveria ser toda 0, mas há {total} não-zero. exemplos: "
                    f"{_coords_as_str(c, coords, end_idx=end_idx, last=last)}",
                )
        else:
            bad = []
            total = 0
            for j in range(m):
                v = c[end_idx, j]
                iv, ok, why = _strict_int(v)
                if (not ok) or (iv != 0):
                    total += 1
                    if len(bad) < max_positions:
                        bad.append((j, v, why))
            if total:
                _add("2", f"crom[-2,:] (linha do end, idx={end_idx}) deveria ser toda 0, mas há {total} inválidos. exemplos(j,valor,motivo)={bad}")

        # 3) crom[-1,-1] == 0
        v = c[last, last]
        iv, ok, why = (_strict_int(v) if not is_int_like else (int(v), True, ""))
        if (not ok) or (iv != 0):
            _add("3", f"crom[-1,-1] deveria ser 0, mas é {v!r} (motivo={why})")

        # Observações estruturais úteis
        v = c[last, 0]
        iv, ok, why = (_strict_int(v) if not is_int_like else (int(v), True, ""))
        if (not ok) or (iv != 0):
            _add("3.obs", f"crom[-1,0] deveria ser 0, mas é {v!r} (motivo={why})")

        v = c[end_idx, last]
        iv, ok, why = (_strict_int(v) if not is_int_like else (int(v), True, ""))
        if (not ok) or (iv != 0):
            _add("3.obs", f"crom[m-2,-1] deveria ser 0, mas é {v!r} (motivo={why})")

        # 4) A só 0/1, e contagens
        row_invalid = np.zeros(A.shape[0], dtype=bool)
        col_invalid = np.zeros(A.shape[1], dtype=bool)

        ones_in_row = np.zeros(A.shape[0], dtype=int)
        ones_in_col = np.zeros(A.shape[1], dtype=int)

        if is_int_like:
            mask_invalid = (A != 0) & (A != 1)
            total_invalid = int(np.count_nonzero(mask_invalid))
            if total_invalid:
                coords = _sample_coords_from_mask(mask_invalid, max_positions)
                _add(
                    "4",
                    f"A=crom[:-1,:-1] deveria ter apenas 0/1, mas há {total_invalid} inválidos. exemplos: "
                    f"{_coords_as_str(c, coords, end_idx=end_idx, last=last)}",
                )
                row_invalid = np.any(mask_invalid, axis=1)
                col_invalid = np.any(mask_invalid, axis=0)

            ones_in_row = np.count_nonzero(A == 1, axis=1).astype(int)
            ones_in_col = np.count_nonzero(A == 1, axis=0).astype(int)
        else:
            total_invalid = 0
            sampled_invalid: List[Tuple[int, int]] = []
            for i in range(A.shape[0]):
                for j in range(A.shape[1]):
                    vv = A[i, j]
                    ivv, okv, _ = _strict_int(vv)
                    if okv and (ivv == 0 or ivv == 1):
                        if ivv == 1:
                            ones_in_row[i] += 1
                            ones_in_col[j] += 1
                    else:
                        total_invalid += 1
                        row_invalid[i] = True
                        col_invalid[j] = True
                        if len(sampled_invalid) < max_positions:
                            sampled_invalid.append((int(i), int(j)))
            if total_invalid:
                _add(
                    "4",
                    f"A=crom[:-1,:-1] deveria ter apenas 0/1, mas há {total_invalid} inválidos. exemplos: "
                    f"{_coords_as_str(c, sampled_invalid, end_idx=end_idx, last=last)}",
                )

        # 5) Domínio dos metadados
        bad_split: List[Tuple[int, Any, str]] = []
        bad_split_total = 0
        for i in range(0, m - 2):  # 0..m-3
            vv = c[i, last]
            ivv, okv, whyv = _strict_int(vv)
            if (not okv) or (ivv not in (-1, 0, 1, 2, 3)):
                bad_split_total += 1
                if len(bad_split) < max_positions:
                    bad_split.append((i, vv, whyv))
        if bad_split_total:
            _add("5", f"split_meta=crom[0..m-3, -1] fora de {{-1,0,1,2,3}}. total={bad_split_total}. exemplos(i,valor,motivo)={bad_split}")

        bad_join: List[Tuple[int, Any, str]] = []
        bad_join_total = 0
        for j in range(1, m - 1):  # 1..m-2
            vv = c[last, j]
            ivv, okv, whyv = _strict_int(vv)
            if (not okv) or (ivv not in (-1, 0, 1, 2, 3)):
                bad_join_total += 1
                if len(bad_join) < max_positions:
                    bad_join.append((j, vv, whyv))
        if bad_join_total:
            _add("5", f"join_meta=crom[-1, 1..m-2] fora de {{-1,0,1,2,3}}. total={bad_join_total}. exemplos(j,valor,motivo)={bad_join}")

        # 5.1) begin/end não podem ter -1
        vv = c[0, last]
        ivv, okv, whyv = _strict_int(vv)
        if (not okv):
            _add("5.1", f"begin: crom[0,-1]={vv!r} inválido ({whyv})")
        elif ivv == -1:
            _add("5.1", "begin: meta_split (crom[0,-1]) não pode ser -1")

        vv = c[last, end_idx]
        ivv, okv, whyv = _strict_int(vv)
        if (not okv):
            _add("5.1", f"end: crom[-1,m-2]={vv!r} inválido ({whyv}). m-2={end_idx}")
        elif ivv == -1:
            _add("5.1", f"end: meta_join (crom[-1,m-2]) não pode ser -1. m-2={end_idx}")

        # 5.2) begin precisa ter pelo menos uma aresta saindo (colunas 1..m-2)
        if is_int_like:
            begin_out = int(np.count_nonzero(A[0, 1:] == 1))
        else:
            begin_out = 0
            for j in range(1, m - 1):
                vv = A[0, j]
                ivv, okv, _ = _strict_int(vv)
                if okv and ivv == 1:
                    begin_out += 1
        if begin_out < 1:
            _add("5.2", f"begin sem aresta saindo: count_ones(A[0, 1..m-2])={begin_out}")

        # 5.3) end precisa ter pelo menos uma aresta chegando (linhas 0..m-3)
        if is_int_like:
            end_in = int(np.count_nonzero(A[:m-2, end_idx] == 1))
        else:
            end_in = 0
            for i in range(0, m - 2):
                vv = A[i, end_idx]
                ivv, okv, _ = _strict_int(vv)
                if okv and ivv == 1:
                    end_in += 1
        if end_in < 1:
            _add("5.3", f"end sem aresta chegando: count_ones(A[0..m-3, m-2])={end_in}. m-2={end_idx}")

        # 5.4–5.6) split_meta por grau (pula linhas contaminadas)
        for i in range(0, m - 2):
            if i < row_invalid.shape[0] and bool(row_invalid[i]):
                continue
            out_count = int(ones_in_row[i]) if i < ones_in_row.shape[0] else 0
            vv = c[i, last]
            ms, ok, _ = _strict_int(vv)
            if not ok:
                continue
            if out_count == 0 and ms != -1:
                _add("5.4", f"i={i} out_count=0, meta_split deveria ser -1, mas é {vv!r}")
            elif out_count == 1 and ms != 0:
                _add("5.5", f"i={i} out_count=1, meta_split deveria ser 0, mas é {vv!r}")
            elif out_count >= 2 and ms not in (1, 2, 3):
                _add("5.6", f"i={i} out_count={out_count}, meta_split deveria ser 1/2/3, mas é {vv!r}")

        # 5.4–5.6) join_meta por grau (pula colunas contaminadas)
        for j in range(1, m - 1):
            if j < col_invalid.shape[0] and bool(col_invalid[j]):
                continue
            in_count = int(ones_in_col[j]) if j < ones_in_col.shape[0] else 0
            vv = c[last, j]
            mj, ok, _ = _strict_int(vv)
            if not ok:
                continue
            if in_count == 0 and mj != -1:
                _add("5.4", f"j={j} in_count=0, meta_join deveria ser -1, mas é {vv!r}")
            elif in_count == 1 and mj != 0:
                _add("5.5", f"j={j} in_count=1, meta_join deveria ser 0, mas é {vv!r}")
            elif in_count >= 2 and mj not in (1, 2, 3):
                _add("5.6", f"j={j} in_count={in_count}, meta_join deveria ser 1/2/3, mas é {vv!r}")

        # 5.7) interno i=1..m-3: split==-1 sse join==-1
        for i in range(1, m - 2):
            ms, ok_ms, why_ms = _strict_int(c[i, last])
            mj, ok_mj, why_mj = _strict_int(c[last, i])
            if not (ok_ms and ok_mj):
                _add("5.7", f"i={i}: meta inválido para comparação: split=({c[i,last]!r},{why_ms}) join=({c[last,i]!r},{why_mj})")
                continue
            if (ms == -1) != (mj == -1):
                _add("5.7", f"i={i}: (meta_split==-1)={(ms==-1)} mas (meta_join==-1)={(mj==-1)}")

        # 6.1) meta_split==0 => A[i,i]==0 (sempre)
        for i in range(0, m - 2):
            ms, ok, _ = _strict_int(c[i, last])
            if not ok or ms != 0:
                continue
            out_count = int(ones_in_row[i])
            if out_count != 1:
                continue  # se 5.5 não vale, não avalia 6.1
            vv = A[i, i]
            dv, okd, whyd = (int(vv), True, "") if is_int_like else _strict_int(vv)
            if (not okd) or dv != 0:
                _add("6.1", f"i={i}: meta_split=0 e out_count=1, mas A[i,i]={vv!r} (motivo={whyd})")

        # 6.2) meta_join==0 => A[j,j]==0 (sempre)
        for j in range(1, m - 1):
            mj, ok, _ = _strict_int(c[last, j])
            if not ok or mj != 0:
                continue

            in_count = int(ones_in_col[j])
            if in_count != 1:
                continue  # se 5.5 não vale, não avalia 6.2

            vv = A[j, j]
            dv, okd, whyd = (int(vv), True, "") if is_int_like else _strict_int(vv)
            if (not okd) or dv != 0:
                _add("6.2", f"j={j}: meta_join=0 e in_count=1, mas A[j,j]={vv!r} (motivo={whyd})")
        if issues:
            _print_report(c)
            return False
        return True

    except Exception as e:
        _add("internal", f"exceção inesperada: {type(e).__name__}: {e}")
        _print_report(cromossome)
        return False
