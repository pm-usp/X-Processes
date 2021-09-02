from random import randint
import cycle as cycle
from multiprocessing import Pool


def nonblank_lines(f):                                                                                                  #?
    for l in f:                                                                                                         #?
        line = l.rstrip()                                                                                               #?
        if line:                                                                                                        #?
            yield line                                                                                                  #?


def plotting_file_creator(var):                                                                                         #?
    plot = open('output-graphs/plotting/plotting_{0}.txt'.format(var), 'w')                                             #?
    plot.close()                                                                                                        #?


def create_plotting_files(number_of_islands, number_of_threads):                                                        #?
    p = Pool(number_of_threads)                                                                                         #?
    p.map(plotting_file_creator, number_of_islands)                                                                     #?
    p.close()                                                                                                           #?


def set_broadcast(population, sorted_evaluated_population, island_number, percentage_of_best_individuals_for_migration_per_island, broadcast):     #?
    all_bests = []                                                                                                      #?
    for i in range(int((len(population)) * percentage_of_best_individuals_for_migration_per_island)):                   #?
        all_bests.append([population[sorted_evaluated_population[i][5]], [sorted_evaluated_population[i][0]]])          #?
    broadcast[island_number] = all_bests                                                                                #?


def pick_island(island_number, number_of_islands, migration_index, broadcast, island_sizes, percentage_of_best_individuals_for_migration_all_islands):     #?
    picked_island = randint(0, number_of_islands - 1)                                                                   #?
    count = 0                                                                                                           #?
    while (picked_island == island_number) or (migration_index[picked_island] >= int(percentage_of_best_individuals_for_migration_all_islands[picked_island] * island_sizes[picked_island])) or (len(broadcast[picked_island]) == 0):      #?
        if count == 10:                                                                                                 #?
            return -1                                                                                                   #?
        else:                                                                                                           #?
            count = count + 1                                                                                           #?
            picked_island = randint(0, number_of_islands - 1)                                                           #?
    return picked_island                                                                                                #?


def send_individuals(population, island_number, number_of_islands, sorted_evaluated_population, messenger):             #?
    picked_island = randint(0, number_of_islands - 1)                                                                   #?
    count = 0                                                                                                           #?
    while picked_island == island_number:                                                                               #?
        if count == 10:                                                                                                 #?
            picked_island = -1                                                                                          #?
        else:                                                                                                           #?
            count = count + 1                                                                                           #?
            picked_island = randint(0, number_of_islands - 1)                                                           #?
    if picked_island != -1:                                                                                             #?
        messenger[picked_island] = [sorted_evaluated_population[0][0], population[sorted_evaluated_population[0][5]], island_number]      #?


def receive_individuals(population, island_number, sorted_evaluated_population, messenger):                             #?
    if messenger[island_number] != [[], [], []]:                                                                        #?
        best_fit = messenger[island_number]                                                                             #?
        messenger[island_number] = [[], [], []]                                                                         #?
    else:                                                                                                               #?
        return 0                                                                                                        #?
    population[sorted_evaluated_population[-1][5]] = best_fit[1]                                                        #?
    return 1                                                                                                            #?


def do_migration(island_population, island_number, number_of_islands, island_fitness, percentage_of_individuals_for_migration_per_island, broadcast, island_sizes, percentage_of_best_individuals_for_migration_all_islands):     #?
    migration_index = []                                                                                                #?
    for i in range(number_of_islands):                                                                                  #?
        migration_index.append(0)                                                                                       #?
    worst_gen_list = []                                                                                                 #?
    count = 0                                                                                                           #?
    for individual in range(len(island_fitness)):                                                                       #?
        worst_gen_list.append([island_fitness[individual], count])                                                      #?
        count += 1                                                                                                      #?
    sorted_worst_gen_list = sorted(worst_gen_list, reverse=False, key=cycle.take_first)                                 #?
    iter = 0                                                                                                            #?
    while iter < percentage_of_individuals_for_migration_per_island * (len(island_population)):                         #?
        picked_island = pick_island(island_number, number_of_islands, migration_index, broadcast, island_sizes, percentage_of_best_individuals_for_migration_all_islands)     #?
        if picked_island != -1:                                                                                         #?
            island_selected = broadcast[picked_island]                                                                  #?
            best_ind = island_selected[migration_index[picked_island]]                                                  #?
            migration_index[picked_island] += 1                                                                         #?
            island_population[sorted_worst_gen_list[iter][1]] = best_ind[0]                                             #?
            iter = iter + 1                                                                                             #?
        else:                                                                                                           #?
            break                                                                                                       #?
    return                                                                                                              #?

















