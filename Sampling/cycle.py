import copy
import operators as opr
import fitness as fit
from operator import itemgetter
import petrinets as pn
import numpy as np


def process_individual_pair(population, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome, alphabet, on_off_task_mutation_probability, max_perc_of_num_tasks_for_on_off_task_mutation, cache_soundness, cache_petri_net, cache_n_tokens, act_fix_mode):
    chosen_individual_1 = opr.parent_selection(evaluated_population[1])
    chosen_individual_2 = opr.parent_selection(evaluated_population[1])
    offspring1 = np.copy(population[chosen_individual_1])
    offspring2 = np.copy(population[chosen_individual_2])
    offspring1, offspring2 = opr.crossover_per_process(crossover_probability, max_perc_of_num_tasks_for_crossover, offspring1, offspring2, reference_cromossome, cache_n_tokens)
    opr.mutation(offspring1, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome, on_off_task_mutation_probability, max_perc_of_num_tasks_for_on_off_task_mutation, cache_n_tokens, act_fix_mode)
    opr.mutation(offspring2, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome, on_off_task_mutation_probability, max_perc_of_num_tasks_for_on_off_task_mutation, cache_n_tokens, act_fix_mode)
    number_of_sound_attempts = 0
    alphabet_len_half = int(len(alphabet) * 0.5)
    sound1 = pn.is_sound(offspring1, alphabet, cache_soundness, cache_petri_net)
    sound2 = pn.is_sound(offspring2, alphabet, cache_soundness, cache_petri_net)
    while (not sound1) or (not sound2):
        offspring1 = np.copy(population[chosen_individual_1])
        offspring2 = np.copy(population[chosen_individual_2])
        if number_of_sound_attempts >= alphabet_len_half:
            break
        else:
            number_of_sound_attempts += 1
            offspring1, offspring2 = opr.crossover_per_process(crossover_probability, max_perc_of_num_tasks_for_crossover, offspring1, offspring2, reference_cromossome, cache_n_tokens)
            opr.mutation(offspring1, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome, on_off_task_mutation_probability, max_perc_of_num_tasks_for_on_off_task_mutation, cache_n_tokens, act_fix_mode)
            opr.mutation(offspring2, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome, on_off_task_mutation_probability, max_perc_of_num_tasks_for_on_off_task_mutation, cache_n_tokens, act_fix_mode)
            sound1 = pn.is_sound(offspring1, alphabet, cache_soundness, cache_petri_net)
            sound2 = pn.is_sound(offspring2, alphabet, cache_soundness, cache_petri_net)
    return offspring1, offspring2


def generation(population, reference_cromossome, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, elitism_percentage, sorted_evaluated_population, alphabet, fitness_weight, precision_weight, generalization_weight, simplicity_weight, simplicity_criterion_methods, input_log_name, round, island, generation, on_off_task_mutation_probability, max_perc_of_num_tasks_for_on_off_task_mutation, cache_fitness, cache_soundness, cache_petri_net, cache_n_tokens, act_fix_mode, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log, filetimestamp):
    auxiliary_population = np.copy(population)
    population_len = len(population)
    for i in range(0, population_len - 1, 2):
        offspring1, offspring2 = process_individual_pair(population, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, reference_cromossome, alphabet, on_off_task_mutation_probability, max_perc_of_num_tasks_for_on_off_task_mutation, cache_soundness, cache_petri_net, cache_n_tokens, act_fix_mode)
        auxiliary_population[i] = offspring1
        auxiliary_population[i + 1] = offspring2
    if population_len % 2 == 1:
        last_index = population_len - 1
        auxiliary_population[last_index - 1] = offspring1
        auxiliary_population[last_index] = offspring2
    if elitism_percentage > 0:
        evaluated_aux_population, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log = fit.evaluate_population(auxiliary_population, alphabet, fitness_weight, precision_weight, generalization_weight, simplicity_weight, simplicity_criterion_methods, input_log_name, cache_fitness, cache_petri_net, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log, filetimestamp)
        sorted_evaluated_aux_population = sorted(evaluated_aux_population[1], reverse=True, key=itemgetter(0, 4, 3))
        opr.elitism(population, elitism_percentage, sorted_evaluated_aux_population, sorted_evaluated_population, auxiliary_population, evaluated_aux_population, evaluated_population)
        evaluated_new_population = evaluated_aux_population
    else:
        evaluated_new_population, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log = fit.evaluate_population(auxiliary_population, alphabet, fitness_weight, precision_weight, generalization_weight, simplicity_weight, simplicity_criterion_methods, input_log_name, cache_fitness, cache_petri_net, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log, filetimestamp)
    return auxiliary_population, evaluated_new_population, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log


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