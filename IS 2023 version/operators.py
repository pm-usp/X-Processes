import random as ran
import copy


def parent_selection(evaluated_population):                                                                             
    opponent_1 = int(ran.random() * len(evaluated_population[1]))                                                       
    opponent_2 = int(ran.random() * len(evaluated_population[1]))
    while opponent_2 == opponent_1:
        opponent_2 = int(ran.random() * len(evaluated_population[1]))
    if evaluated_population[1][opponent_1][0] >= evaluated_population[1][opponent_2][0]:
        if evaluated_population[1][opponent_1][0] > evaluated_population[1][opponent_2][0]:
            return opponent_1
        else:
            if evaluated_population[1][opponent_1][4] >= evaluated_population[1][opponent_2][4]:
                if evaluated_population[1][opponent_1][4] > evaluated_population[1][opponent_2][4]:
                    return opponent_1
                else:
                    if evaluated_population[1][opponent_1][3] >= evaluated_population[1][opponent_2][3]:
                        return opponent_1
                    else:
                        return opponent_2
            else:
                return opponent_2
    else:                                                                                                               
        return opponent_2                                                                                               


def crossover_per_process(crossover_probability, max_perc_of_num_tasks_for_crossover, cromossome_1, cromossome_2):
    if ran.random() < crossover_probability:
        while True:
            unreachableOrNoProperEndTask = True
            offspring_1 = copy.deepcopy(cromossome_1)                                                                       
            offspring_2 = copy.deepcopy(cromossome_2)                                                                       
            if max_perc_of_num_tasks_for_crossover == -1:                                                                       
                number_of_tasks = 1                                                                                         
            else:                                                                                                           
                max_number_of_tasks = int(max_perc_of_num_tasks_for_crossover * (len(cromossome_1) - 1))
                if max_number_of_tasks < 1:
                    max_number_of_tasks = 1
                number_of_tasks = ran.randint(1, max_number_of_tasks)
            for i in range(number_of_tasks):
                chosen_task = int(ran.random() * (len(cromossome_1) - 1))                                                   
                offspring_1[chosen_task] = cromossome_2[chosen_task]                                                        
                offspring_2[chosen_task] = cromossome_1[chosen_task]                                                        
                for j in range(len(offspring_1)):                                                                           
                    offspring_1[j][chosen_task] = cromossome_2[j][chosen_task]                                              
                    offspring_2[j][chosen_task] = cromossome_1[j][chosen_task]
            number_of_produced_XOR_tokens = count_produced_XOR_tokens(offspring_1)
            number_of_consumed_XOR_tokens = count_consumed_XOR_tokens(offspring_1)
            number_of_produced_AND_tokens = count_produced_AND_tokens(offspring_1)
            number_of_consumed_AND_tokens = count_consumed_AND_tokens(offspring_1)
            if (number_of_consumed_XOR_tokens != number_of_produced_XOR_tokens) or (number_of_consumed_AND_tokens != number_of_produced_AND_tokens):
                if ran.random() < 0.5:
                    increase_number_of_produced_tokens(offspring_1, number_of_consumed_XOR_tokens, number_of_consumed_AND_tokens)
                else:
                    increase_number_of_consumed_tokens(offspring_1, number_of_produced_XOR_tokens, number_of_produced_AND_tokens)
            number_of_produced_XOR_tokens = count_produced_XOR_tokens(offspring_2)
            number_of_consumed_XOR_tokens = count_consumed_XOR_tokens(offspring_2)
            number_of_produced_AND_tokens = count_produced_AND_tokens(offspring_2)
            number_of_consumed_AND_tokens = count_consumed_AND_tokens(offspring_2)
            if (number_of_consumed_XOR_tokens != number_of_produced_XOR_tokens) or (number_of_consumed_AND_tokens != number_of_produced_AND_tokens):
                if ran.random() < 0.5:
                    increase_number_of_produced_tokens(offspring_2, number_of_consumed_XOR_tokens, number_of_consumed_AND_tokens)
                else:
                    increase_number_of_consumed_tokens(offspring_2, number_of_produced_XOR_tokens, number_of_produced_AND_tokens)

            if unreachableOrNoProperEndTask == True:
                for n in range(len(offspring_1) - 2):
                    if is_there_at_least_one_raw_active_task(offspring_1, n) == False:
                        unreachableOrNoProperEndTask = False
                        break
            if unreachableOrNoProperEndTask == True:
                for n in range(1, len(offspring_1) - 1):
                    if is_there_at_least_one_column_active_task(offspring_1, n) == False:
                        unreachableOrNoProperEndTask = False
                        break
            if unreachableOrNoProperEndTask == True:
                for n in range(len(offspring_2) - 2):
                    if is_there_at_least_one_raw_active_task(offspring_2, n) == False:
                        unreachableOrNoProperEndTask = False
                        break
            if unreachableOrNoProperEndTask == True:
                for n in range(1, len(offspring_2) - 1):
                    if is_there_at_least_one_column_active_task(offspring_2, n) == False:
                        unreachableOrNoProperEndTask = False
                        break
            if unreachableOrNoProperEndTask == True:
                return offspring_2, offspring_1

    return cromossome_1, cromossome_2                                                                                   


def mutation(cromossome, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome):
    if ran.random() < task_mutation_probability:
        task_mutation(cromossome, max_perc_of_num_tasks_for_task_mutation, reference_cromossome)
    if ran.random() < gateway_mutation_probability:
        gateway_mutation(cromossome, max_perc_of_num_tasks_for_gateway_mutation)
    return
                                                                                                                

def task_mutation(cromossome, max_perc_of_num_tasks_for_task_mutation, reference_cromossome):
    while True:
        temp_cromossome = copy.deepcopy(cromossome)
        unreachableOrNoProperEndTask = True
        if max_perc_of_num_tasks_for_task_mutation == -1:
            number_of_tasks = 1
        else:
            max_number_of_tasks = int(max_perc_of_num_tasks_for_task_mutation * (len(cromossome) - 1))
            if max_number_of_tasks < 1:
                max_number_of_tasks = 1
            number_of_tasks = ran.randint(1, max_number_of_tasks)
        for i in range(number_of_tasks):
            chosen_task = int(ran.random() * (len(cromossome) - 1))
            chosen_mutation_type = ran.random()
            if chosen_mutation_type < 1/2:
                for j in range(len(cromossome)):
                    new_input = int(ran.random() * (len(cromossome) - 1))
                    if reference_cromossome[new_input][chosen_task] == 1:
                        break
                if cromossome[new_input][chosen_task] == 0:
                    cromossome[new_input][chosen_task] = 1
                else:
                    cromossome[new_input][chosen_task] = 0
                    for k in range(1, len(cromossome) - 1):
                        if cromossome[new_input][k] == 1:
                            break
                    if k == len(cromossome) - 2:
                        chosen_task2 = int(ran.random() * (len(cromossome) - 1))
                        count = 0
                        while (chosen_task2 == chosen_task) or (chosen_task2 == new_input) or (reference_cromossome[new_input][chosen_task2] == 0):
                            chosen_task2 = int(ran.random() * (len(cromossome) - 1))
                            count += 1
                            if count > (len(cromossome) * 0.25):
                                chosen_task2 = chosen_task
                                break
                        cromossome[new_input][chosen_task2] = 1
                    for k in range(len(cromossome) - 2):
                        if cromossome[k][chosen_task] == 1:
                            break
                    if k == len(cromossome) - 3:
                        new_input2 = int(ran.random() * (len(cromossome) - 2))
                        count = 0
                        while (new_input2 == new_input) or (new_input2 == chosen_task) or (reference_cromossome[new_input2][chosen_task] == 0):
                            new_input2 = int(ran.random() * (len(cromossome) - 1))
                            count += 1
                            if count > (len(cromossome) * 0.25):
                                new_input2 = new_input
                                break
                        cromossome[new_input2][chosen_task] = 1
                    if cromossome[new_input][chosen_task] == 0:
                        if cromossome[new_input][-1] == 1:
                            if is_there_at_least_two_raw_active_tasks(cromossome, new_input) == False:
                                cromossome[new_input][-1] = 0
                        if cromossome[-1][chosen_task] == 1:
                            if is_there_at_least_two_column_active_tasks(cromossome, chosen_task) == False:
                                cromossome[-1][chosen_task] = 0
                number_of_produced_XOR_tokens = count_produced_XOR_tokens(cromossome)
                number_of_consumed_XOR_tokens = count_consumed_XOR_tokens(cromossome)
                number_of_produced_AND_tokens = count_produced_AND_tokens(cromossome)
                number_of_consumed_AND_tokens = count_consumed_AND_tokens(cromossome)
                if (number_of_consumed_XOR_tokens != number_of_produced_XOR_tokens) or (number_of_consumed_AND_tokens != number_of_produced_AND_tokens):
                    if ran.random() < 0.5:
                        increase_number_of_produced_tokens(cromossome, number_of_consumed_XOR_tokens, number_of_consumed_AND_tokens)
                    else:
                        increase_number_of_consumed_tokens(cromossome, number_of_produced_XOR_tokens, number_of_produced_AND_tokens)
            else:
                for j in range(len(cromossome)):
                    new_output = int(ran.random() * (len(cromossome) - 1))
                    if reference_cromossome[chosen_task][new_output] == 1:
                        break
                if cromossome[chosen_task][new_output] == 0:
                    cromossome[chosen_task][new_output] = 1
                else:
                    cromossome[chosen_task][new_output] = 0
                    for k in range(1, len(cromossome) - 1):
                        if cromossome[chosen_task][k] == 1:
                            break
                    if k == len(cromossome) - 2:
                        new_output2 = int(ran.random() * (len(cromossome) - 1))
                        count = 0
                        while (new_output2 == new_output) or (new_output2 == chosen_task) or (reference_cromossome[chosen_task][new_output2] == 0):
                            new_output2 = int(ran.random() * (len(cromossome) - 1))
                            count += 1
                            if count > (len(cromossome) * 0.25):
                                new_output2 = new_output
                                break
                        cromossome[chosen_task][new_output2] = 1
                    for k in range(len(cromossome) - 2):
                        if cromossome[k][new_output] == 1:
                            break
                    if k == len(cromossome) - 3:
                        chosen_task2 = int(ran.random() * (len(cromossome) - 1))
                        count = 0
                        while (chosen_task2 == chosen_task) or (chosen_task2 == new_output) or (reference_cromossome[chosen_task2][new_output] == 0):
                            chosen_task2 = int(ran.random() * (len(cromossome) - 1))
                            count += 1
                            if count > (len(cromossome) * 0.25):
                                chosen_task2 = chosen_task
                                break
                        cromossome[chosen_task2][new_output] = 1
                    if cromossome[chosen_task][new_output] == 0:
                        if cromossome[chosen_task][-1] == 1:
                            if is_there_at_least_two_raw_active_tasks(cromossome, chosen_task) == False:
                                cromossome[chosen_task][-1] = 0
                        if cromossome[-1][new_output] == 1:
                            if is_there_at_least_two_column_active_tasks(cromossome, new_output) == False:
                                cromossome[-1][new_output] = 0
                number_of_produced_XOR_tokens = count_produced_XOR_tokens(cromossome)
                number_of_consumed_XOR_tokens = count_consumed_XOR_tokens(cromossome)
                number_of_produced_AND_tokens = count_produced_AND_tokens(cromossome)
                number_of_consumed_AND_tokens = count_consumed_AND_tokens(cromossome)
                if (number_of_consumed_XOR_tokens != number_of_produced_XOR_tokens) or (number_of_consumed_AND_tokens != number_of_produced_AND_tokens):
                    if ran.random() < 0.5:
                        increase_number_of_produced_tokens(cromossome, number_of_consumed_XOR_tokens, number_of_consumed_AND_tokens)
                    else:
                        increase_number_of_consumed_tokens(cromossome, number_of_produced_XOR_tokens, number_of_produced_AND_tokens)
        if unreachableOrNoProperEndTask == True:
            for n in range(len(cromossome) - 2):
                if is_there_at_least_one_raw_active_task(cromossome, n) == False:
                    unreachableOrNoProperEndTask = False
                    break
        if unreachableOrNoProperEndTask == True:
            for n in range(1, len(cromossome) - 1):
                if is_there_at_least_one_column_active_task(cromossome, n) == False:
                    unreachableOrNoProperEndTask = False
                    break
        if unreachableOrNoProperEndTask == True:
            return
        else:
            cromossome = copy.deepcopy(temp_cromossome)


def gateway_mutation(cromossome, max_perc_of_num_tasks_for_gateway_mutation):
    if max_perc_of_num_tasks_for_gateway_mutation == -1:
        number_of_tasks = 1
    else:
        max_number_of_tasks = int(max_perc_of_num_tasks_for_gateway_mutation * (len(cromossome) - 1))
        if max_number_of_tasks < 1:
            max_number_of_tasks = 1
        number_of_tasks = ran.randint(1, max_number_of_tasks)
    for i in range(number_of_tasks):
        chosen_operator_position = ran.random()
        if chosen_operator_position < 1/2:
            chosen_task = int(ran.random() * (len(cromossome) - 2))
            if cromossome[chosen_task][-1] == 0:
                if is_there_at_least_two_raw_active_tasks(cromossome, chosen_task):
                    cromossome[chosen_task][-1] = 1
            else:
                cromossome[chosen_task][-1] = 0
            number_of_produced_XOR_tokens = count_produced_XOR_tokens(cromossome)
            number_of_consumed_XOR_tokens = count_consumed_XOR_tokens(cromossome)
            number_of_produced_AND_tokens = count_produced_AND_tokens(cromossome)
            number_of_consumed_AND_tokens = count_consumed_AND_tokens(cromossome)
            if (number_of_consumed_XOR_tokens != number_of_produced_XOR_tokens) or (number_of_consumed_AND_tokens != number_of_produced_AND_tokens):
                if ran.random() < 0.5:
                    increase_number_of_produced_tokens(cromossome, number_of_consumed_XOR_tokens, number_of_consumed_AND_tokens)
                else:
                    increase_number_of_consumed_tokens(cromossome, number_of_produced_XOR_tokens, number_of_produced_AND_tokens)
        else:
            chosen_task = int(ran.random() * (len(cromossome) - 2)) + 1
            if cromossome[-1][chosen_task] == 0:
                if is_there_at_least_two_column_active_tasks(cromossome, chosen_task):
                    cromossome[-1][chosen_task] = 1
            else:
                cromossome[-1][chosen_task] = 0
            number_of_produced_XOR_tokens = count_produced_XOR_tokens(cromossome)
            number_of_consumed_XOR_tokens = count_consumed_XOR_tokens(cromossome)
            number_of_produced_AND_tokens = count_produced_AND_tokens(cromossome)
            number_of_consumed_AND_tokens = count_consumed_AND_tokens(cromossome)
            if (number_of_consumed_XOR_tokens != number_of_produced_XOR_tokens) or (number_of_consumed_AND_tokens != number_of_produced_AND_tokens):
                if ran.random() < 0.5:
                    increase_number_of_produced_tokens(cromossome, number_of_consumed_XOR_tokens, number_of_consumed_AND_tokens)
                else:
                    increase_number_of_consumed_tokens(cromossome, number_of_produced_XOR_tokens, number_of_produced_AND_tokens)
    return


def count_produced_XOR_tokens(cromossome):
    number_of_produced_XOR_tokens = 0
    for i in range(len(cromossome) - 2):
        if cromossome[i][-1] == 0:
            number_of_produced_XOR_tokens = number_of_produced_XOR_tokens + 1
    return number_of_produced_XOR_tokens


def count_consumed_XOR_tokens(cromossome):
    number_of_consumed_XOR_tokens = 0
    for i in range(1, len(cromossome) - 1):
        if cromossome[-1][i] == 0:
            number_of_consumed_XOR_tokens = number_of_consumed_XOR_tokens + 1
    return number_of_consumed_XOR_tokens


def count_produced_AND_tokens(cromossome):
    number_of_produced_AND_tokens = 0
    for i in range(len(cromossome) - 2):
        if cromossome[i][-1] == 1:
            for j in range(1, len(cromossome[i]) - 1):
                number_of_produced_AND_tokens = number_of_produced_AND_tokens + cromossome[i][j]
    return number_of_produced_AND_tokens


def count_consumed_AND_tokens(cromossome):
    number_of_consumed_AND_tokens = 0
    for i in range(1, len(cromossome) - 1):
        if cromossome[-1][i] == 1:
            for j in range(len(cromossome[i]) - 2):
                number_of_consumed_AND_tokens = number_of_consumed_AND_tokens + cromossome[j][i]
    return number_of_consumed_AND_tokens


def increase_number_of_produced_tokens(cromossome, number_of_consumed_XOR_tokens, number_of_consumed_AND_tokens):
    copy_of_output_gateways = []
    for i in range(len(cromossome)):
        copy_of_output_gateways.append(cromossome[i][-1])
    number_of_attempts = int(len(cromossome) * 0.5)
    for i in range(number_of_attempts):
        chosen_task = ran.randint(0, int(len(cromossome) - 3))
        max_i = 0
        while (cromossome[chosen_task][-1] == 1) and (max_i <= (len(cromossome) * 2)):
            max_i = max_i + 1
            chosen_task = ran.randint(0, int(len(cromossome) - 3))
        if is_there_at_least_two_raw_active_tasks(cromossome, chosen_task):
            cromossome[chosen_task][-1] = 1
        number_of_produced_XOR_tokens = count_produced_XOR_tokens(cromossome)
        number_of_produced_AND_tokens = count_produced_AND_tokens(cromossome)
        max_i = 0
        while ((not((number_of_consumed_XOR_tokens == number_of_produced_XOR_tokens) and (number_of_consumed_AND_tokens == number_of_produced_AND_tokens))) and (max_i <= (len(cromossome) * 2))):
            max_i = max_i + 1
            chosen_task = ran.randint(0, int(len(cromossome) - 3))
            max_j = 0
            while (cromossome[chosen_task][-1] == 1) and (max_j <= (len(cromossome) * 2)):
                max_j = max_j + 1
                chosen_task = ran.randint(0, int(len(cromossome) - 3))
            if is_there_at_least_two_raw_active_tasks(cromossome, chosen_task):
                cromossome[chosen_task][-1] = 1
            number_of_produced_XOR_tokens = count_produced_XOR_tokens(cromossome)
            number_of_produced_AND_tokens = count_produced_AND_tokens(cromossome)
        if (number_of_consumed_XOR_tokens != number_of_produced_XOR_tokens) or (number_of_consumed_AND_tokens != number_of_produced_AND_tokens):
            for j in range(len(cromossome)):
                cromossome[j][-1] = copy.deepcopy(copy_of_output_gateways[j])
    if i == number_of_attempts - 1:
        decrease_number_of_consumed_tokens(cromossome, count_produced_XOR_tokens(cromossome), count_produced_AND_tokens(cromossome))
    return


def increase_number_of_consumed_tokens(cromossome, number_of_produced_XOR_tokens, number_of_produced_AND_tokens):
    copy_of_input_gateways = copy.deepcopy(cromossome[-1])
    number_of_attempts = int(len(cromossome) * 0.5)
    for i in range(number_of_attempts):
        chosen_task = ran.randint(1, int(len(cromossome) - 2))
        max_i = 0
        while (cromossome[-1][chosen_task] == 1) and (max_i <= (len(cromossome) * 2)):
            max_i = max_i + 1
            chosen_task = ran.randint(1, int(len(cromossome) - 2))
        if is_there_at_least_two_column_active_tasks(cromossome, chosen_task):
            cromossome[-1][chosen_task] = 1
        number_of_consumed_XOR_tokens = count_consumed_XOR_tokens(cromossome)
        number_of_consumed_AND_tokens = count_consumed_AND_tokens(cromossome)
        max_i = 0
        while ((not((number_of_consumed_XOR_tokens == number_of_produced_XOR_tokens) and (number_of_consumed_AND_tokens == number_of_produced_AND_tokens))) and (max_i <= (len(cromossome) * 2))):
            max_i = max_i + 1
            chosen_task = ran.randint(1, int(len(cromossome) - 2))
            max_j = 0
            while (cromossome[-1][chosen_task] == 1) and (max_j <= (len(cromossome) * 2)):
                max_j = max_j + 1
                chosen_task = ran.randint(1, int(len(cromossome) - 2))
            if is_there_at_least_two_column_active_tasks(cromossome, chosen_task):
                cromossome[-1][chosen_task] = 1
            number_of_consumed_XOR_tokens = count_consumed_XOR_tokens(cromossome)
            number_of_consumed_AND_tokens = count_consumed_AND_tokens(cromossome)
        if (number_of_consumed_XOR_tokens != number_of_produced_XOR_tokens) or (number_of_consumed_AND_tokens != number_of_produced_AND_tokens):
            cromossome[-1] = copy.deepcopy(copy_of_input_gateways)
    if i == number_of_attempts - 1:
        decrease_number_of_produced_tokens(cromossome, count_consumed_XOR_tokens(cromossome), count_consumed_AND_tokens(cromossome))
    return


def decrease_number_of_produced_tokens(cromossome, number_of_consumed_XOR_tokens, number_of_consumed_AND_tokens):
    copy_of_output_gateways = []
    for i in range(len(cromossome)):
        copy_of_output_gateways.append(cromossome[i][-1])
    number_of_attempts = int(len(cromossome) * 0.5)
    for i in range(number_of_attempts):
        chosen_task = ran.randint(0, int(len(cromossome) - 3))
        max_i = 0
        while (cromossome[chosen_task][-1] == 0) and (max_i <= (len(cromossome) * 2)):
            max_i = max_i + 1
            chosen_task = ran.randint(0, int(len(cromossome) - 3))
        cromossome[chosen_task][-1] = 0
        number_of_produced_XOR_tokens = count_produced_XOR_tokens(cromossome)
        number_of_produced_AND_tokens = count_produced_AND_tokens(cromossome)
        max_i = 0
        while ((not((number_of_consumed_XOR_tokens == number_of_produced_XOR_tokens) and (number_of_consumed_AND_tokens == number_of_produced_AND_tokens))) and (max_i <= (len(cromossome) * 2))):
            max_i = max_i + 1
            chosen_task = ran.randint(0, int(len(cromossome) - 3))
            max_j = 0
            while (cromossome[chosen_task][-1] == 0) and (max_j <= (len(cromossome) * 2)):
                max_j = max_j + 1
                chosen_task = ran.randint(0, int(len(cromossome) - 3))
            cromossome[chosen_task][-1] = 0
            number_of_produced_XOR_tokens = count_produced_XOR_tokens(cromossome)
            number_of_produced_AND_tokens = count_produced_AND_tokens(cromossome)
        if (number_of_consumed_XOR_tokens != number_of_produced_XOR_tokens) or (number_of_consumed_AND_tokens != number_of_produced_AND_tokens):
            for j in range(len(cromossome)):
                cromossome[j][-1] = copy.deepcopy(copy_of_output_gateways[j])
    return


def decrease_number_of_consumed_tokens(cromossome, number_of_produced_XOR_tokens, number_of_produced_AND_tokens):
    copy_of_input_gateways = copy.deepcopy(cromossome[-1])
    number_of_attempts = int(len(cromossome) * 0.5)
    for i in range(number_of_attempts):
        chosen_task = ran.randint(1, int(len(cromossome) - 2))
        max_i = 0
        while (cromossome[-1][chosen_task] == 0) and (max_i <= (len(cromossome) * 2)):
            max_i = max_i + 1
            chosen_task = ran.randint(1, int(len(cromossome) - 2))
        cromossome[-1][chosen_task] = 0
        number_of_consumed_XOR_tokens = count_consumed_XOR_tokens(cromossome)
        number_of_consumed_AND_tokens = count_consumed_AND_tokens(cromossome)
        max_i = 0
        while ((not((number_of_consumed_XOR_tokens == number_of_produced_XOR_tokens) and (number_of_consumed_AND_tokens == number_of_produced_AND_tokens))) and (max_i <= (len(cromossome) * 2))):
            max_i = max_i + 1
            chosen_task = ran.randint(1, int(len(cromossome) - 2))
            max_j = 0
            while (cromossome[-1][chosen_task] == 0) and (max_j <= (len(cromossome) * 2)):
                max_j = max_j + 1
                chosen_task = ran.randint(1, int(len(cromossome) - 2))
            cromossome[-1][chosen_task] = 0
            number_of_consumed_XOR_tokens = count_consumed_XOR_tokens(cromossome)
            number_of_consumed_AND_tokens = count_consumed_AND_tokens(cromossome)
        if (number_of_consumed_XOR_tokens != number_of_produced_XOR_tokens) or (number_of_consumed_AND_tokens != number_of_produced_AND_tokens):
            cromossome[-1] = copy.deepcopy(copy_of_input_gateways)
    return


def is_there_at_least_one_raw_active_task(cromossome, chosen_task):
    number_of_active_tasks = 0
    for j in range(1, len(cromossome[chosen_task]) - 1):
        if cromossome[chosen_task][j] == 1:
            number_of_active_tasks += 1
        if number_of_active_tasks == 1:
            break
    if number_of_active_tasks == 1:
        return True
    else:
        return False


def is_there_at_least_one_column_active_task(cromossome, chosen_task):
    number_of_active_tasks = 0
    for j in range(len(cromossome[chosen_task]) - 2):
        if cromossome[j][chosen_task] == 1:
            number_of_active_tasks += 1
        if number_of_active_tasks == 1:
            break
    if number_of_active_tasks == 1:
        return True
    else:
        return False


def is_there_at_least_two_raw_active_tasks(cromossome, chosen_task):
    number_of_active_tasks = 0
    for j in range(1, len(cromossome[chosen_task]) - 1):
        if cromossome[chosen_task][j] == 1:
            number_of_active_tasks += 1
        if number_of_active_tasks == 2:
            break
    if number_of_active_tasks == 2:
        return True
    else:
        return False


def is_there_at_least_two_column_active_tasks(cromossome, chosen_task):
    number_of_active_tasks = 0
    for j in range(len(cromossome[chosen_task]) - 2):
        if cromossome[j][chosen_task] == 1:
            number_of_active_tasks += 1
        if number_of_active_tasks == 2:
            break
    if number_of_active_tasks == 2:
        return True
    else:
        return False


def elitism(population, elitism_percentage, sorted_evaluated_aux_population, sorted_evaluated_population, auxiliary_population, evaluated_aux_population, evaluated_population):
    for i in range(round(len(population) * elitism_percentage)):
        if sorted_evaluated_aux_population[len(sorted_evaluated_aux_population) - 1 - i][0] < sorted_evaluated_population[i][0]:
            auxiliary_population[sorted_evaluated_aux_population[len(sorted_evaluated_aux_population) - 1 - i][5]] = copy.deepcopy(population[sorted_evaluated_population[i][5]])

            evaluated_aux_population[0] = evaluated_aux_population[0] - evaluated_aux_population[1][sorted_evaluated_aux_population[len(sorted_evaluated_aux_population) - 1 - i][5]][0]
            aux_list = list(evaluated_aux_population[1][sorted_evaluated_aux_population[len(sorted_evaluated_aux_population) - 1 - i][5]])
            aux_list[0] = evaluated_population[1][sorted_evaluated_population[i][5]][0]
            aux_list[1] = evaluated_population[1][sorted_evaluated_population[i][5]][1]
            aux_list[2] = evaluated_population[1][sorted_evaluated_population[i][5]][2]
            aux_list[3] = evaluated_population[1][sorted_evaluated_population[i][5]][3]
            aux_list[4] = evaluated_population[1][sorted_evaluated_population[i][5]][4]
            evaluated_aux_population[1][sorted_evaluated_aux_population[len(sorted_evaluated_aux_population) - 1 - i][5]] = tuple(aux_list)
            evaluated_aux_population[0] = evaluated_aux_population[0] + evaluated_aux_population[1][sorted_evaluated_aux_population[len(sorted_evaluated_aux_population) - 1 - i][5]][0]

        else:
            break                                                                                                                                                               
    return
