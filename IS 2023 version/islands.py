from random import randint
import cycle as cycle
from multiprocessing import Pool


def nonblank_lines(f):                                                                                                  
    for l in f:                                                                                                         
        line = l.rstrip()                                                                                               
        if line:                                                                                                        
            yield line                                                                                                  


def plotting_file_creator(var):                                                                                         
    plot = open('output-graphs/plotting/plotting_{0}.txt'.format(var), 'w')                                             
    plot.close()                                                                                                        


def create_plotting_files(number_of_islands, number_of_threads):                                                        
    p = Pool(number_of_threads)                                                                                         
    p.map(plotting_file_creator, number_of_islands)                                                                     
    p.close()                                                                                                           


def set_broadcast(population, sorted_evaluated_population, island_number, percentage_of_best_individuals_for_migration_per_island, broadcast):     
    all_bests = []                                                                                                      
    for i in range(int((len(population)) * percentage_of_best_individuals_for_migration_per_island)):                   
        all_bests.append([population[sorted_evaluated_population[i][5]], [sorted_evaluated_population[i][0]], [sorted_evaluated_population[i][1]], [sorted_evaluated_population[i][2]], [sorted_evaluated_population[i][3]], [sorted_evaluated_population[i][4]], [sorted_evaluated_population[i][5]], [sorted_evaluated_population[i][6]]])          
    broadcast[island_number] = all_bests                                                                                


def set_broadcast_null(island_number, broadcast):     
    broadcast[island_number] = []                                                                                                      


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
        messenger[picked_island] = [sorted_evaluated_population[0][0], population[sorted_evaluated_population[0][5]], island_number, [sorted_evaluated_population[0][0]], [sorted_evaluated_population[0][1]], [sorted_evaluated_population[0][2]], [sorted_evaluated_population[0][3]], [sorted_evaluated_population[0][4]], [sorted_evaluated_population[0][5]], [sorted_evaluated_population[0][6]]]


def send_individuals_null(island_number, messenger):             
    messenger[island_number] = [[], [], []]


def receive_individuals(population, island_number, sorted_evaluated_population, messenger, evaluated_population):                             
    if messenger[island_number] != [[], [], []]:                                                                        
        best_fit = messenger[island_number]                                                                             
        messenger[island_number] = [[], [], []]                                                                         
    else:                                                                                                               
        return 0                                                                                                        
    if sorted_evaluated_population[-1][0] < best_fit[0]:
        population[sorted_evaluated_population[-1][5]] = best_fit[1]                                                        
        evaluated_population[0] = evaluated_population[0] - evaluated_population[1][sorted_evaluated_population[-1][5]][0]
        aux_list = list(evaluated_population[1][sorted_evaluated_population[-1][5]])
        aux_list[0] = best_fit[3][0]
        aux_list[1] = best_fit[4][0]
        aux_list[2] = best_fit[5][0]
        aux_list[3] = best_fit[6][0]
        aux_list[4] = best_fit[7][0]
        evaluated_population[1][sorted_evaluated_population[-1][5]] = tuple(aux_list)
        evaluated_population[0] = evaluated_population[0] + evaluated_population[1][sorted_evaluated_population[-1][5]][0]
        return 1                                                                                                            
    else:
        return 0


def do_migration(island_population, island_number, number_of_islands, island_fitness, percentage_of_individuals_for_migration_per_island, broadcast, island_sizes, percentage_of_best_individuals_for_migration_all_islands, evaluated_population):     
    migration_index = []                                                                                                
    for i in range(number_of_islands):                                                                                  
        migration_index.append(0)                                                                                       
    worst_gen_list = []                                                                                                 
    count = 0                                                                                                           
    for individual in range(len(island_fitness)):                                                                       
        worst_gen_list.append([island_fitness[individual], count])                                                      
        count += 1                                                                                                      
    sorted_worst_gen_list = sorted(worst_gen_list, reverse=False, key=cycle.take_first)                                 
    iter = 0                                                                                                            
    while iter < percentage_of_individuals_for_migration_per_island * (len(island_population)):                         
        picked_island = pick_island(island_number, number_of_islands, migration_index, broadcast, island_sizes, percentage_of_best_individuals_for_migration_all_islands)     
        if picked_island != -1:                                                                                         
            island_selected = broadcast[picked_island]                                                                  
            best_ind = island_selected[migration_index[picked_island]]                                                  
            migration_index[picked_island] += 1                                                                         
            if evaluated_population[1][sorted_worst_gen_list[iter][1]][0] < best_ind[1][0]:
                island_population[sorted_worst_gen_list[iter][1]] = best_ind[0]                                             
                evaluated_population[0] = evaluated_population[0] - evaluated_population[1][sorted_worst_gen_list[iter][1]][0]
                aux_list = list(evaluated_population[1][sorted_worst_gen_list[iter][1]])
                aux_list[0] = best_ind[1][0]
                aux_list[1] = best_ind[2][0]
                aux_list[2] = best_ind[3][0]
                aux_list[3] = best_ind[4][0]
                aux_list[4] = best_ind[5][0]
                evaluated_population[1][sorted_worst_gen_list[iter][1]] = tuple(aux_list)
                evaluated_population[0] = evaluated_population[0] + evaluated_population[1][sorted_worst_gen_list[iter][1]][0]
                iter = iter + 1                                                                                             
        else:                                                                                                           
            break                                                                                                       
    return                                                                                                              

















