import tm
from random import randint
from multiprocessing import Pool

@tm.measure_time
def nonblank_lines(f):
    for l in f:                                                                                                         
        line = l.rstrip()                                                                                               
        if line:                                                                                                        
            yield line                                                                                                  

@tm.measure_time
def plotting_file_creator(var):
    plot = open('output-graphs/plotting/plotting_{0}.txt'.format(var), 'w')                                             
    plot.close()                                                                                                        

@tm.measure_time
def create_plotting_files(number_of_islands, number_of_threads):
    p = Pool(number_of_threads)                                                                                         
    p.map(plotting_file_creator, number_of_islands)                                                                     
    p.close()                                                                                                           

@tm.measure_time
def set_broadcast(population, sorted_evaluated_population, island_number, percentage_of_best_individuals_for_migration_per_island, broadcast):
    all_bests = []
    for i in range(int((len(population)) * percentage_of_best_individuals_for_migration_per_island)):
        all_bests.append([population[int(sorted_evaluated_population[i][5])], [sorted_evaluated_population[i][0]], [sorted_evaluated_population[i][1]], [sorted_evaluated_population[i][2]], [sorted_evaluated_population[i][3]], [sorted_evaluated_population[i][4]], [sorted_evaluated_population[i][5]], [sorted_evaluated_population[i][6]]])
    broadcast[island_number] = all_bests

@tm.measure_time
def set_broadcast_null(island_number, broadcast):
    broadcast[island_number] = []

@tm.measure_time
def pick_island(island_number, number_of_islands, migration_index, broadcast, island_sizes, percentage_of_best_individuals_for_migration_all_islands):
    picked_island = randint(0, number_of_islands - 1)
    count = 0
    while (picked_island == island_number) or (migration_index[picked_island] >= int(percentage_of_best_individuals_for_migration_all_islands[picked_island] * island_sizes[picked_island])) or (len(broadcast[picked_island]) == 0):
        if count == 10:
            return -1
        else:
            count = count + 1
            picked_island = randint(0, number_of_islands - 1)
    return picked_island

@tm.measure_time
def send_individuals(population, island_number, number_of_islands, sorted_evaluated_population, messenger):
    picked_island = randint(0, number_of_islands - 1)
    count = 0
    while picked_island == island_number:
        if count == 10:
            picked_island = -1
        else:
            count = count + 1
            picked_island = randint(0, number_of_islands - 1)
    if picked_island != -1:
        messenger[picked_island] = [sorted_evaluated_population[0][0], population[int(sorted_evaluated_population[0][5])], island_number, [sorted_evaluated_population[0][0]], [sorted_evaluated_population[0][1]], [sorted_evaluated_population[0][2]], [sorted_evaluated_population[0][3]], [sorted_evaluated_population[0][4]], [sorted_evaluated_population[0][5]], [sorted_evaluated_population[0][6]]]

@tm.measure_time
def send_individuals_null(island_number, messenger):
    messenger[island_number] = [[], [], []]

@tm.measure_time
def receive_individuals(population, island_number, sorted_evaluated_population, messenger, evaluated_population):
    if messenger[island_number] != [[], [], []]:
        best_fit = messenger[island_number]
        messenger[island_number] = [[], [], []]
    else:
        return 0
    if sorted_evaluated_population[-1][0] < best_fit[0]:
        population[int(sorted_evaluated_population[-1][5])] = best_fit[1]
        evaluated_population[0] = evaluated_population[0] - evaluated_population[1][int(sorted_evaluated_population[-1][5])][0]
        aux_list = list(evaluated_population[1][int(sorted_evaluated_population[-1][5])])
        aux_list[0] = best_fit[3][0]
        aux_list[1] = best_fit[4][0]
        aux_list[2] = best_fit[5][0]
        aux_list[3] = best_fit[6][0]
        aux_list[4] = best_fit[7][0]
        evaluated_population[1][int(sorted_evaluated_population[-1][5])] = tuple(aux_list)
        evaluated_population[0] = evaluated_population[0] + evaluated_population[1][int(sorted_evaluated_population[-1][5])][0]
        return 1
    else:
        return 0

@tm.measure_time
def do_migration(island_population, island_number, number_of_islands, island_fitness, percentage_of_individuals_for_migration_per_island, broadcast, island_sizes, percentage_of_best_individuals_for_migration_all_islands, evaluated_population):
    migration_index = [0] * number_of_islands
    worst_gen_list = [(fitness, i) for i, fitness in enumerate(island_fitness)]
    worst_gen_list.sort(key=lambda x: x[0])
    total_migrations = int(percentage_of_individuals_for_migration_per_island * len(island_population))
    iter = 0
    while iter < total_migrations:
        picked_island = pick_island(island_number, number_of_islands, migration_index, broadcast, island_sizes, percentage_of_best_individuals_for_migration_all_islands)
        if picked_island == -1:
            break
        island_selected = broadcast[picked_island]
        best_ind = island_selected[migration_index[picked_island]]
        migration_index[picked_island] += 1
        worst_ind_index = worst_gen_list[iter][1]
        worst_ind_fitness = evaluated_population[1][worst_ind_index][0]
        if worst_ind_fitness < best_ind[1][0]:
            island_population[worst_ind_index] = best_ind[0]
            evaluated_population[0] -= worst_ind_fitness
            aux_list = list(evaluated_population[1][worst_ind_index])
            aux_list[0] = best_ind[1][0]
            aux_list[1] = best_ind[2][0]
            aux_list[2] = best_ind[3][0]
            aux_list[3] = best_ind[4][0]
            aux_list[4] = best_ind[5][0]
            evaluated_population[1][worst_ind_index] = tuple(aux_list)
            evaluated_population[0] += aux_list[0]
            iter += 1
    return

# @tm.measure_time
# def nonblank_lines_tm(f):
#     return nonblank_lines(f)
#
# @tm.measure_time
# def plotting_file_creator_tm(var):
#     return plotting_file_creator(var)
#
# @tm.measure_time
# def create_plotting_files_tm(number_of_islands, number_of_threads):
#     return create_plotting_files(number_of_islands, number_of_threads)
#
# @tm.measure_time
# def set_broadcast_tm(population, sorted_evaluated_population, island_number, percentage_of_best_individuals_for_migration_per_island, broadcast):
#     return set_broadcast(population, sorted_evaluated_population, island_number, percentage_of_best_individuals_for_migration_per_island, broadcast)
#
# @tm.measure_time
# def set_broadcast_null_tm(island_number, broadcast):
#     return set_broadcast_null(island_number, broadcast)
#
# @tm.measure_time
# def pick_island_tm(island_number, number_of_islands, migration_index, broadcast, island_sizes, percentage_of_best_individuals_for_migration_all_islands):
#     return pick_island(island_number, number_of_islands, migration_index, broadcast, island_sizes, percentage_of_best_individuals_for_migration_all_islands)
#
# @tm.measure_time
# def send_individuals_tm(population, island_number, number_of_islands, sorted_evaluated_population, messenger):
#     return send_individuals(population, island_number, number_of_islands, sorted_evaluated_population, messenger)
#
# @tm.measure_time
# def send_individuals_null_tm(island_number, messenger):
#     return send_individuals_null(island_number, messenger)
#
# @tm.measure_time
# def receive_individuals_tm(population, island_number, sorted_evaluated_population, messenger, evaluated_population):
#     return receive_individuals(population, island_number, sorted_evaluated_population, messenger, evaluated_population)
#
# @tm.measure_time
# def do_migration_tm(island_population, island_number, number_of_islands, island_fitness, percentage_of_individuals_for_migration_per_island, broadcast, island_sizes, percentage_of_best_individuals_for_migration_all_islands, evaluated_population):
#     return do_migration(island_population, island_number, number_of_islands, island_fitness, percentage_of_individuals_for_migration_per_island, broadcast, island_sizes, percentage_of_best_individuals_for_migration_all_islands, evaluated_population)
