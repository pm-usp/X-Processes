import plotting as plt
import recording as rec
import cycle as cyc
import initialpop as inp
import fitness as fit
import islands as isl
import multiprocessing as mlp
import datetime as dt
import log
from ast import literal_eval
from functools import partial
import petrinets as pn


number_of_islands = 4
number_of_rounds = 1
max_number_of_generations = 5000
max_processing_time = 24 * 60 * 60

max_perc_of_num_tasks_for_crossover = 0.7
crossover_probability = 0.7
task_mutation_probability = 0.7
gateway_mutation_probability = 0.7
max_perc_of_num_tasks_for_task_mutation = 0.5
max_perc_of_num_tasks_for_gateway_mutation = 0.3
elitism_percentage = 0.3

stop_condition = 5


def run_round(paramenter_set, number_of_islands, round, broadcast, messenger, island_sizes, percentage_of_best_individuals_for_migration_all_islands, full_log, alphabet, log_name, xes_log, max_number_of_generations, max_processing_time, max_perc_of_num_tasks_for_crossover, crossover_probability, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, elitism_percentage, stop_condition, island_number):     #?
    island_start = dt.datetime.now()                                                                                    
    population_size = int(paramenter_set[island_number + 1][0])                                                         
    migration_time = int(paramenter_set[island_number + 1][1])                                                          
    percentage_of_best_individuals_for_migration_per_island = float(paramenter_set[island_number + 1][2])               
    percentage_of_individuals_for_migration_per_island = float(paramenter_set[island_number + 1][3])                    
    island_sizes[island_number] = population_size                                                                       
    percentage_of_best_individuals_for_migration_all_islands[island_number] = percentage_of_best_individuals_for_migration_per_island     
    (population, evaluated_population, reference_cromossome, average_enabled_tasks) = inp.initialize_population(population_size, alphabet, full_log, xes_log)     #?
    fitness_evolution = []                                                                                              #?
    (highest_value_and_position, sorted_evaluated_population) = cyc.choose_highest(evaluated_population)                #?
    lowest_value = cyc.choose_lowest(sorted_evaluated_population)                                                       #?
    average_value = cyc.calculate_average(evaluated_population)                                                         #?
    fitness_evolution.append([lowest_value, highest_value_and_position[0][0], average_value, 0, highest_value_and_position[0][1], highest_value_and_position[0][2], highest_value_and_position[0][3], highest_value_and_position[0][4], 0, 0, 0, 0])     #?
    pn.create_and_show_pn(population[highest_value_and_position[1]], alphabet, island_number, 0 , log_name, 0)
    island_end = dt.datetime.now()                                                                                      #?
    island_duration = island_end - island_start                                                                         #?
    print('RND:', round, '| F-SCORE:', '%.3f' % highest_value_and_position[0][0], '| FIT:', '%.3f' % highest_value_and_position[0][1], '| PREC:', '%.3f' % highest_value_and_position[0][2], '| GEN:', '%.3f' % highest_value_and_position[0][3], '| SIMP:', '%.3f' % highest_value_and_position[0][4], '| ISL:', '{:>2}'.format(island_number), '| GEN:', '{:>3}'.format('0'), '| REP:', '{:>2}'.format(fitness_evolution[0][8]), '{:>2}'.format(fitness_evolution[0][3]), '{:>2}'.format(fitness_evolution[0][9]), '{:>2}'.format(fitness_evolution[0][10]), '{:>2}'.format(fitness_evolution[0][11]), '|', island_duration)     #?
    for current_generation in range(1, max_number_of_generations):                                                          #?
        (population, evaluated_population) = cyc.generation(population, reference_cromossome, evaluated_population, crossover_probability, max_perc_of_num_tasks_for_crossover, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, elitism_percentage, sorted_evaluated_population, alphabet, xes_log)     #?
        (highest_value_and_position, sorted_evaluated_population) = cyc.choose_highest(evaluated_population)            #?
        lowest_value = cyc.choose_lowest(sorted_evaluated_population)                                                   #?
        average_value = cyc.calculate_average(evaluated_population)                                                     #?
        fitness_evolution.append([lowest_value, highest_value_and_position[0][0], average_value, 0, highest_value_and_position[0][1], highest_value_and_position[0][2], highest_value_and_position[0][3], highest_value_and_position[0][4], 0, 0, 0, 0])     #?
        if fitness_evolution[current_generation][1] == fitness_evolution[current_generation - 1][1]:                    #?
            fitness_evolution[current_generation][8] = fitness_evolution[current_generation - 1][8] + 1                 #?
        else:                                                                                                           #?
             if fitness_evolution[current_generation][1] < fitness_evolution[current_generation - 1][1]:                #?
                fitness_evolution[current_generation][8] = -1                                                           #?
        if fitness_evolution[current_generation][4] == fitness_evolution[current_generation - 1][4]:                    #?
            fitness_evolution[current_generation][3] = fitness_evolution[current_generation - 1][3] + 1                 #?
        else:                                                                                                           #?
            if fitness_evolution[current_generation][4] < fitness_evolution[current_generation - 1][4]:                 #?
                fitness_evolution[current_generation][3] = -1                                                           #?
        if fitness_evolution[current_generation][5] == fitness_evolution[current_generation - 1][5]:                    #?
            fitness_evolution[current_generation][9] = fitness_evolution[current_generation - 1][9] + 1                 #?
        else:                                                                                                           #?
            if fitness_evolution[current_generation][5] < fitness_evolution[current_generation - 1][5]:                 #?
                fitness_evolution[current_generation][9] = -1                                                           #?
        if fitness_evolution[current_generation][6] == fitness_evolution[current_generation - 1][6]:                    #?
            fitness_evolution[current_generation][10] = fitness_evolution[current_generation - 1][10] + 1               #?
        else:                                                                                                           #?
            if fitness_evolution[current_generation][6] < fitness_evolution[current_generation - 1][6]:                 #?
                fitness_evolution[current_generation][10] = -1                                                          #?
        if fitness_evolution[current_generation][7] == fitness_evolution[current_generation - 1][7]:                    #?
            fitness_evolution[current_generation][11] = fitness_evolution[current_generation - 1][11] + 1               #?
        else:                                                                                                           #?
            if fitness_evolution[current_generation][7] < fitness_evolution[current_generation - 1][7]:                 #?
                fitness_evolution[current_generation][11] = -1                                                          #?
        pn.create_and_show_pn(population[highest_value_and_position[1]], alphabet, island_number, current_generation, log_name, round)
        island_end = dt.datetime.now()                                                                                  #?
        island_duration = island_end - island_start                                                                     #?
        print('RND:', round, '| F-SCORE:', '%.3f' % highest_value_and_position[0][0], '| FIT:', '%.3f' % highest_value_and_position[0][1], '| PREC:', '%.3f' % highest_value_and_position[0][2], '| GEN:', '%.3f' % highest_value_and_position[0][3], '| SIMP:', '%.3f' % highest_value_and_position[0][4], '| ISL:', '{:>2}'.format(island_number), '| GEN:', '{:>3}'.format(current_generation), '| REP:', '{:>2}'.format(fitness_evolution[current_generation][8]), '{:>2}'.format(fitness_evolution[current_generation][3]), '{:>2}'.format(fitness_evolution[current_generation][9]), '{:>2}'.format(fitness_evolution[current_generation][10]), '{:>2}'.format(fitness_evolution[current_generation][11]), '|', island_duration)     #?
        isl.set_broadcast(population, sorted_evaluated_population, island_number, percentage_of_best_individuals_for_migration_per_island, broadcast)     #?
        if ((fitness_evolution[current_generation][8] >= stop_condition)) or (island_duration > dt.timedelta(seconds=max_processing_time)):     #?
            break                                                                                                       #?
        if isl.receive_individuals(population, island_number, sorted_evaluated_population, messenger) == 1:             #?
            evaluated_population = fit.evaluate_population(population, alphabet, xes_log)                               #?
            (highest_value_and_position, sorted_evaluated_population) = cyc.choose_highest(evaluated_population)        #?
        if (current_generation > 0) and (current_generation % migration_time == 0):                                     #?
            island_fitness = []                                                                                         #?
            for i in range(len(evaluated_population[1])):                                                               #?
                island_fitness.append(evaluated_population[1][i][0])                                                    #?
            isl.do_migration(population, island_number, number_of_islands, island_fitness, percentage_of_individuals_for_migration_per_island, broadcast, island_sizes, percentage_of_best_individuals_for_migration_all_islands)     #?
            isl.send_individuals(population, island_number, number_of_islands, sorted_evaluated_population, messenger)  #?
            evaluated_population = fit.evaluate_population(population, alphabet, xes_log)                               #?
            (highest_value_and_position, sorted_evaluated_population) = cyc.choose_highest(evaluated_population)        #?
        rec.record_evolution(log_name, str(round), paramenter_set[island_number + 1], island_number, highest_value_and_position[0], fitness_evolution, alphabet, population[highest_value_and_position[1]], island_start, island_end, island_duration, current_generation)     #?
    plt.plot_evolution_per_island(fitness_evolution, log_name, str(round), island_number)                       #?
    previous_plotting = []                                                                                              #?
    with open('output-graphs/plotting/plotting_{0}.txt'.format(island_number), 'r') as plott:                           #?
        for line in isl.nonblank_lines(plott):                                                                          #?
            previous_plotting.append(literal_eval(line))                                                                #?
    plott.close()                                                                                                       #?
    previous_plotting.extend(fitness_evolution)                                                                         #?
    with open('output-graphs/plotting/plotting_{0}.txt'.format(island_number), 'w') as plott:                           #?
        for ini in range(len(previous_plotting)):                                                                       #?
            plott.write(str(previous_plotting[ini]) + '\n')                                                             #?
    plott.close()                                                                                                       #?
    rec.record_evolution(log_name, str(round), paramenter_set[island_number + 1], island_number, highest_value_and_position[0], fitness_evolution, alphabet, population[highest_value_and_position[1]], island_start, island_end, island_duration, current_generation)     #?
    print('Final results ==>', 'ISL:', island_number, '| ISL-DURANTION:', island_duration, '| F-SCORE:', '%.10f' % highest_value_and_position[0][0], '| FIT:', '%.10f' % highest_value_and_position[0][1], '| PREC:', '%.10f' % highest_value_and_position[0][2], '| GEN:', '%.10f' % highest_value_and_position[0][3], '| SIMP:', '%.10f' % highest_value_and_position[0][4], '| ALPHABET:', alphabet, '| BEST INDIVIDUAL:', population[highest_value_and_position[1]])     #?
    print('===================================================')                                                        #?
    return                                                                                                              #?


if __name__ == '__main__':                                                                                              #?
    for log_index in range(1):
        for round in range(0, number_of_rounds):
            paramenter_set = []
            alphabet = []
            island_ids = []
            with open('input-parameters/input-parameters.csv', 'r') as parameters:
                for line in isl.nonblank_lines(parameters):
                    paramenter_set.append(line.split(';'))
            parameters.close()
            full_log, log_name, xes_log = log.import_log(log_index)                                                                  #?
            inp.create_alphabet(full_log, alphabet)                                                                         #?
            inp.process_log(full_log)
            for island in range(number_of_islands):                                                                         #?
                island_ids.append(island)                                                                                   #?
            isl.create_plotting_files(island_ids, number_of_islands)                                                        #?
            p = mlp.Pool(number_of_islands)                                                                                 #?
            m = mlp.Manager()                                                                                               #?
            broadcast = m.list()                                                                                            #?
            messenger = m.list()                                                                                            #?
            island_sizes = m.list()                                                                                         #?
            percentage_of_best_individuals_for_migration_all_islands = m.list()                                             #?
            func = partial(run_round, paramenter_set, number_of_islands)                                                    #?
            for island in range(number_of_islands):                                                                         #?
                broadcast.append([])                                                                                        #?
                messenger.append([[],[],[]])                                                                                #?
                island_sizes.append(0)                                                                                      #?
                percentage_of_best_individuals_for_migration_all_islands.append(0)                                          #?
            broadcast.append(1)                                                                                             #?
            func2 = partial(func, round, broadcast, messenger, island_sizes, percentage_of_best_individuals_for_migration_all_islands, full_log, alphabet, log_name, xes_log, max_number_of_generations, max_processing_time, max_perc_of_num_tasks_for_crossover, crossover_probability, task_mutation_probability, gateway_mutation_probability, max_perc_of_num_tasks_for_task_mutation, max_perc_of_num_tasks_for_gateway_mutation, elitism_percentage, stop_condition)     #?
            p.map(func2, island_ids)                                                                                        #?
            plt.plot_evolution_integrated(log_name, str(round), number_of_islands)                                  #?
            p.close()                                                                                                           #?