import copy
import operators as opr
import fitness as fit
from operator import itemgetter
import petrinets as pn


def generation(population, reference_cromossome, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, elitism_percentage, sorted_evaluated_population, alphabet, xes_log):     #?
    auxiliary_population = copy.deepcopy(population)                                                                    #?
    i = 0                                                                                                               #?
    while i < len(population):                                                                                          #?
        chosen_individual_1 = opr.parent_selection(evaluated_population)                                                #?
        auxiliary_population[i] = copy.deepcopy(population[chosen_individual_1])                                        #?
        chosen_individual_2 = opr.parent_selection(evaluated_population)                                                #?
        auxiliary_population[i + 1] = copy.deepcopy(population[chosen_individual_2])                                    #?
        (auxiliary_population[i], auxiliary_population[i + 1]) = opr.crossover_per_process(crossover_probability, max_perc_of_num_tasks_for_crossover, auxiliary_population[i], auxiliary_population[i+1])     #?
        opr.mutation(auxiliary_population[i], task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome)                     #?
        opr.mutation(auxiliary_population[i + 1], task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome)                   #?
        number_of_sound_attempts = 0
        while not pn.is_sound(auxiliary_population[i], alphabet) or not pn.is_sound(auxiliary_population[i + 1], alphabet):
            auxiliary_population[i] = copy.deepcopy(population[chosen_individual_1])
            auxiliary_population[i + 1] = copy.deepcopy(population[chosen_individual_2])
            if number_of_sound_attempts == int(len(alphabet) * 0.5):
                break
            else:
                number_of_sound_attempts += 1
                (auxiliary_population[i], auxiliary_population[i + 1]) = opr.crossover_per_process(crossover_probability, max_perc_of_num_tasks_for_crossover, auxiliary_population[i], auxiliary_population[i + 1])  # ?
                opr.mutation(auxiliary_population[i], task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome)  # ?
                opr.mutation(auxiliary_population[i + 1], task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome)  # ?
        i = i + 2                                                                                                       #?
        if (i + 1 == len(population)) and (len(population) % 2 == 1):                                                   #?
            i = i - 1                                                                                                   #?
    if elitism_percentage > 0:                                                                                        #?
        evaluated_aux_population = fit.evaluate_population(auxiliary_population, alphabet, xes_log)                     #?
        sorted_evaluated_aux_population = sorted(evaluated_aux_population[1], reverse=True, key=itemgetter(0,4,3))             #?                                                                                      #?
        opr.elitism(population, elitism_percentage, sorted_evaluated_aux_population, sorted_evaluated_population, auxiliary_population)     #?
    evaluated_new_population = fit.evaluate_population(auxiliary_population, alphabet, xes_log)                         #?
    return (auxiliary_population, evaluated_new_population)                                                             #?


def take_first(elem):                                                                                                   #?
    return elem[0]                                                                                                      #?


def choose_highest(evaluated_population):                                                                               #?
    highest_value = [-1, -1, -1, -1, -1]                                                                                #?
    sorted_evaluated_population = sorted(evaluated_population[1], reverse=True, key=take_first)                         #?
    highest_value[0] = sorted_evaluated_population[0][0]                                                                #?
    highest_value[1] = sorted_evaluated_population[0][1]                                                                #?
    highest_value[2] = sorted_evaluated_population[0][2]                                                                #?
    highest_value[3] = sorted_evaluated_population[0][3]                                                                #?
    highest_value[4] = sorted_evaluated_population[0][4]                                                                #?
    highest_position = sorted_evaluated_population[0][5]                                                                #?
    return (highest_value, highest_position), sorted_evaluated_population                                               #?


def choose_lowest(sorted_evaluated_population):                                                                         #?
    return sorted_evaluated_population[-1][0]                                                                           #?


def calculate_average(evaluated_population):                                                                            #?
    value_sum = 0                                                                                                       #?
    for i in range(len(evaluated_population[1])):                                                                       #?
        value_sum = value_sum + evaluated_population[1][i][0]                                                           #?
    return value_sum / len(evaluated_population[1])                                                                     #?