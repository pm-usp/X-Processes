import copy
import operators as opr
import fitness as fit
from operator import itemgetter
import petrinets as pn
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_individual_pair(individual_pair, population, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome, alphabet, log_name, round, island, generation):
    chosen_individual_1 = opr.parent_selection(evaluated_population)
    chosen_individual_2 = opr.parent_selection(evaluated_population)
    offspring1 = np.copy(population[chosen_individual_1])
    offspring2 = np.copy(population[chosen_individual_2])
    offspring1, offspring2 = opr.crossover_per_process(crossover_probability, max_perc_of_num_tasks_for_crossover, offspring1, offspring2)
    opr.mutation(offspring1, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome)
    opr.mutation(offspring2, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome)
    number_of_sound_attempts = 0
    alphabet_len_half = int(len(alphabet) * 0.5)
    while not pn.is_sound(offspring1, alphabet, log_name, round, island, generation) or not pn.is_sound(offspring2, alphabet, log_name, round, island, generation):
        offspring1 = np.copy(population[chosen_individual_1])
        offspring2 = np.copy(population[chosen_individual_2])
        if number_of_sound_attempts >= alphabet_len_half:
            break
        else:
            number_of_sound_attempts += 1
            offspring1, offspring2 = opr.crossover_per_process(crossover_probability, max_perc_of_num_tasks_for_crossover, offspring1, offspring2)
            opr.mutation(offspring1, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome)
            opr.mutation(offspring2, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome)
    return individual_pair, offspring1, offspring2

def generation(population, reference_cromossome, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, elitism_percentage, sorted_evaluated_population, alphabet, xes_log, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight, log_name, round, island, generation):
    auxiliary_population = np.copy(population)
    population_len = len(population)
    for i in range(0, population_len - 1, 2):
        individual_pair, offspring1, offspring2 = process_individual_pair((i, i + 1), population, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome, alphabet, log_name, round, island, generation)
        auxiliary_population[i] = offspring1
        auxiliary_population[i + 1] = offspring2
    if population_len % 2 == 1:
        last_index = population_len - 1
        individual_pair, offspring1, offspring2 = process_individual_pair((last_index - 1, last_index), population, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome, alphabet, log_name, round, island, generation)
        auxiliary_population[last_index - 1] = offspring1
        auxiliary_population[last_index] = offspring2
    if elitism_percentage > 0:
        evaluated_aux_population = fit.evaluate_population(auxiliary_population, alphabet, xes_log, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight, log_name, round, island, generation)
        elite_count = int(elitism_percentage * population_len)
        sorted_evaluated_aux_population = sorted(evaluated_aux_population[1][:elite_count], reverse=True, key=itemgetter(0, 4, 3))
        opr.elitism(population, elitism_percentage, sorted_evaluated_aux_population, sorted_evaluated_population, auxiliary_population, evaluated_aux_population, evaluated_population)
        evaluated_new_population = copy.deepcopy(evaluated_aux_population)
    else:
        evaluated_new_population = fit.evaluate_population(auxiliary_population, alphabet, xes_log, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight, log_name, round, island, generation)
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