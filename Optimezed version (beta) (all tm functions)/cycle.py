import tm
import copy
import operators as opr
import fitness as fit
from operator import itemgetter
import petrinets as pn
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_individual_pair(individual_pair, population, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome, alphabet):
    chosen_individual_1 = opr.parent_selection_tm(evaluated_population)
    chosen_individual_2 = opr.parent_selection_tm(evaluated_population)
    offspring1 = np.copy(population[chosen_individual_1])
    offspring2 = np.copy(population[chosen_individual_2])
    offspring1, offspring2 = opr.crossover_per_process_tm(crossover_probability, max_perc_of_num_tasks_for_crossover, offspring1, offspring2)
    opr.mutation_tm(offspring1, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome)
    opr.mutation_tm(offspring2, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome)
    number_of_sound_attempts = 0
    alphabet_len_half = int(len(alphabet) * 0.5)
    while not pn.is_sound(offspring1, alphabet) or not pn.is_sound(offspring2, alphabet):
        offspring1 = np.copy(population[chosen_individual_1])
        offspring2 = np.copy(population[chosen_individual_2])
        if number_of_sound_attempts >= alphabet_len_half:
            break
        else:
            number_of_sound_attempts += 1
            offspring1, offspring2 = opr.crossover_per_process_tm(crossover_probability, max_perc_of_num_tasks_for_crossover, offspring1, offspring2)
            opr.mutation_tm(offspring1, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome)
            opr.mutation_tm(offspring2, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome)

    return individual_pair, offspring1, offspring2

def generation(population, reference_cromossome, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, elitism_percentage, sorted_evaluated_population, alphabet, xes_log, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight):
    auxiliary_population = np.copy(population)
    population_len = len(population)

    # with ThreadPoolExecutor() as executor:
    #     futures = []
    #     for i in range(0, population_len - 1, 2):
    #         futures.append(executor.submit(process_individual_pair, (i, i + 1), population, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome, alphabet))
    #     if population_len % 2 == 1:
    #         last_index = population_len - 1
    #         futures.append(executor.submit(process_individual_pair, (last_index - 1, last_index), population, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome, alphabet))
    #     for future in as_completed(futures):
    #         individual_pair, offspring1, offspring2 = future.result()
    #         i, i_plus_1 = individual_pair
    #         auxiliary_population[i] = offspring1
    #         auxiliary_population[i_plus_1] = offspring2

    for i in range(0, population_len - 1, 2):
        individual_pair, offspring1, offspring2 = process_individual_pair((i, i + 1), population, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome, alphabet)
        auxiliary_population[i] = offspring1
        auxiliary_population[i + 1] = offspring2
    if population_len % 2 == 1:
        last_index = population_len - 1
        individual_pair, offspring1, offspring2 = process_individual_pair((last_index - 1, last_index), population, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome, alphabet)
        auxiliary_population[last_index - 1] = offspring1
        auxiliary_population[last_index] = offspring2

    if elitism_percentage > 0:
        evaluated_aux_population = fit.evaluate_population(auxiliary_population, alphabet, xes_log, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight)
        elite_count = int(elitism_percentage * population_len)
        sorted_evaluated_aux_population = sorted(evaluated_aux_population[1][:elite_count], reverse=True, key=itemgetter(0, 4, 3))
        opr.elitism_tm(population, elitism_percentage, sorted_evaluated_aux_population, sorted_evaluated_population, auxiliary_population, evaluated_aux_population, evaluated_population)
        evaluated_new_population = copy.deepcopy(evaluated_aux_population)
    else:
        evaluated_new_population = fit.evaluate_population(auxiliary_population, alphabet, xes_log, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight)

    return auxiliary_population, evaluated_new_population

def take_first(elem):
    return elem[0]                                                                                                      

def choose_highest(evaluated_population):                                                                               
    highest_value = [-1, -1, -1, -1, -1]                                                                                
    sorted_evaluated_population = sorted(evaluated_population[1], reverse=True, key=take_first)
    highest_value[0] = sorted_evaluated_population[0][0]
    highest_value[1] = sorted_evaluated_population[0][1]                                                                
    highest_value[2] = sorted_evaluated_population[0][2]                                                                
    highest_value[3] = sorted_evaluated_population[0][3]                                                                
    highest_value[4] = sorted_evaluated_population[0][4]                                                                
    highest_position = sorted_evaluated_population[0][5]

    return (highest_value, highest_position), sorted_evaluated_population                                               

def choose_lowest(sorted_evaluated_population):                                                                         
    return sorted_evaluated_population[-1][0]                                                                           

def calculate_average(evaluated_population):
    value_sum = 0                                                                                                       
    for i in range(len(evaluated_population[1])):                                                                       
        value_sum += evaluated_population[1][i][0]

    return value_sum / len(evaluated_population[1])

#Time measurement approch (Guilherme)
@tm.measure_time
def process_individual_pair_tm(individual_pair, population, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome, alphabet):
    return process_individual_pair(individual_pair, population, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome, alphabet)

@tm.measure_time
def generation_tm(population, reference_cromossome, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, elitism_percentage, sorted_evaluated_population, alphabet, xes_log, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight):
    return generation(population, reference_cromossome, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, elitism_percentage, sorted_evaluated_population, alphabet, xes_log, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight)
@tm.measure_time
def take_first_tm(elem):
    return take_first(elem)
@tm.measure_time
def choose_highest_tm(evaluated_population):
    return choose_highest(evaluated_population)
@tm.measure_time
def choose_lowest_tm(sorted_evaluated_population):
    return choose_lowest(sorted_evaluated_population)
@tm.measure_time
def calculate_average_tm(evaluated_population):
    return calculate_average(evaluated_population)