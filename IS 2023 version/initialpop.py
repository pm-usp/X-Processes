import random as ran
import fitness as fit
import petrinets as pn
import operators as op

def create_alphabet(log, alphabet):
    for i in range(len(log)):                                                                                           
        for j in range(len(log[i])):                                                                                    
            if alphabet.count(log[i][j]) == 0:                                                                          
                alphabet.append(log[i][j])                                                                              
    alphabet.sort()
    alphabet.insert(0,'begin')
    alphabet.append('end')
    print('Number of activities:', len(alphabet) - 2)
    return


def process_log(log):
    for i in range(len(log)):
        log[i].insert(0, 'begin')
        log[i].append('end')
    return


def initialize_population(island_number, population_size, alphabet, log, xes_log, algo_option, SOLVER_LIMIT):
    population = [initialize_individual(len(alphabet)) for _ in range(population_size)]                                 #TROCAR POR NUMPY? ARRAY do numpy. Ver na biblioteca de genéticos do python qual é a estrutura que ele usa.
    (reference_cromossome, average_enabled_tasks) = create_DFG(log, alphabet)
    for i in range(len(population)):
        population[i] = create_initial_individual(island_number, population[i], alphabet, reference_cromossome)
        while not pn.is_sound(population[i], alphabet):
            population[i] = initialize_individual(len(alphabet))
            population[i] = create_initial_individual(island_number, population[i], alphabet, reference_cromossome)
    population[0] = reference_cromossome
    return (population, fit.evaluate_population(island_number, population, alphabet, xes_log, algo_option, SOLVER_LIMIT), reference_cromossome, average_enabled_tasks)


def initialize_individual(number_of_tasks):
    individual = [create_empty_individual_task(number_of_tasks) for _ in range(number_of_tasks + 1)]                    
    return individual                                                                                                   


def create_empty_individual_task(number_of_tasks):                                                                      
    task = [0 for _ in range(number_of_tasks + 1)]                                                                      
    return task                                                                                                         


def create_DFG(log, alphabet):                                                                         
    DFG = [create_empty_individual_task(len(alphabet)) for _ in range(len(alphabet) + 1)]              
    for i in range(len(alphabet)):                                                                                      
        for j in range(len(log)):                                                                                       
            for k in range(len(log[j]) - 1):                                                                            
                if log[j][k] == alphabet[i]:                                                                            
                    DFG[i][get_task_id(log[j][k + 1], alphabet)] = 1                                   
    enabled_tasks = 0                                                                                                   
    for j in range(len(DFG) - 1):                                                                      
        for k in range(1, len(DFG[j]) - 1):                                                            
            if DFG[j][k] == 1:                                                                         
                enabled_tasks = enabled_tasks + 1                                                                       
    return (DFG, enabled_tasks)                                                                        


def get_task_id(task, alphabet):                                                                                        
    return alphabet.index(task)                                                                                         


def create_initial_individual(island_number, cromossome, alphabet, reference_cromossome):
    #create tasks
    for n in range(ran.randint(1, int(len(alphabet) * 1))):
        x1 = ran.randint(1, len(alphabet) - 2)
        while reference_cromossome[0][x1] == 0:
            x1 = ran.randint(1, len(alphabet) - 2)
        cromossome[0][x1] = 1
        end_achieved = False
        for i in range(1, len(alphabet) - 2):
            x2 = x1
            x1 = ran.randint(1, len(alphabet) - 1)
            while reference_cromossome[x2][x1] == 0:
                x1 = ran.randint(1, len(alphabet) - 1)
            cromossome[x2][x1] = 1
            if x1 == len(alphabet) - 1:
                end_achieved = True
                break
        if end_achieved == False:
            while reference_cromossome[x1][-2] == 0:
                x2 = x1
                x1 = ran.randint(1, len(alphabet) - 1)
                while reference_cromossome[x2][x1] == 0:
                    x1 = ran.randint(1, len(alphabet) - 1)
                cromossome[x2][x1] = 1
                if x1 == len(alphabet) - 1:
                    end_achieved = True
                    break
            if end_achieved == False:
                cromossome[x1][-2] = 1
    for n in range(len(cromossome) - 2):
        if op.is_there_at_least_one_raw_active_task(cromossome, n) == False:
            chosen_task = ran.randint(1, len(cromossome) - 2)
            while reference_cromossome[n][chosen_task] == 0:
                chosen_task = ran.randint(1, len(cromossome) - 2)
            cromossome[n][chosen_task] = 1
    for n in range(1, len(cromossome) - 1):
        if op.is_there_at_least_one_column_active_task(cromossome, n) == False:
            chosen_task = ran.randint(0, len(cromossome) - 3)
            while reference_cromossome[chosen_task][n] == 0:
                chosen_task = ran.randint(0, len(cromossome) - 3)
            cromossome[chosen_task][n] = 1

    #create gateways (AND)
    for n in range(len(cromossome) - 2):
        if ran.random() < ran.randint(0, 1)/100:
            cromossome[n][-1] = 1
    for n in range(1, len(cromossome) - 1):
        if ran.random() < ran.randint(0, 1)/100:
            cromossome[-1][n] = 1
    number_of_produced_XOR_tokens = op.count_produced_XOR_tokens(cromossome)
    number_of_consumed_XOR_tokens = op.count_consumed_XOR_tokens(cromossome)
    number_of_produced_AND_tokens = op.count_produced_AND_tokens(cromossome)
    number_of_consumed_AND_tokens = op.count_consumed_AND_tokens(cromossome)
    if (number_of_consumed_XOR_tokens != number_of_produced_XOR_tokens) or (number_of_consumed_AND_tokens != number_of_produced_AND_tokens):
        if ran.random() < 0.5:
            op.increase_number_of_produced_tokens(cromossome, number_of_consumed_XOR_tokens, number_of_consumed_AND_tokens)
        else:
            op.increase_number_of_consumed_tokens(cromossome, number_of_produced_XOR_tokens, number_of_produced_AND_tokens)
    return cromossome
