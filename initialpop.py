import random as ran
import fitness as fit
from pm4py.algo.analysis.woflan import algorithm as woflan
import petrinets as pn
import operators as op
import itertools


test_sound = [
    [0, 1, 1, 1, 0, 0, 1, 1, 0, 0],
    [1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 0, 0, 0, 0, 1, 0],
    [0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 1, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]]

test_unsound = [
    [0, 0, 0, 1, 0, 0, 1, 1, 0, 1],
    [1, 1, 1, 1, 0, 1, 1, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 0, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 0, 1, 0, 1, 0, 0],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 1, 0, 1, 0, 1, 1, 0, 0],
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 0],
    [1, 0, 0, 0, 1, 0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]


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


def initialize_population(population_size, alphabet, log, xes_log):
    population = [initialize_individual(len(alphabet)) for _ in range(population_size)]                                 #TROCAR POR NUMPY? ARRAY do numpy. Ver na biblioteca de genéticos do python qual é a estrutura que ele usa.
    (reference_cromossome, average_enabled_tasks) = create_DFG(log, alphabet)                          
    for i in range(len(population)):                                                                                    
        population[i] = create_initial_individual2(population[i], alphabet, reference_cromossome)
        #population[i] = reference_cromossome
        while not pn.is_sound(population[i], alphabet):
            population[i] = initialize_individual(len(alphabet))
            population[i] = create_initial_individual2(population[i], alphabet, reference_cromossome)
            #population[i] = reference_cromossome
    population[0] = reference_cromossome
    #for i in range(len(population)): #usado para teste com uma população inicial específica
        #population[i] = copy.deepcopy(test_unsound)
    return (population, fit.evaluate_population(population, alphabet, xes_log), reference_cromossome, average_enabled_tasks)     


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


def create_initial_individual(cromossome, alphabet, log):                                                               
    influence_control = 1 #control the "influence" of the dependency measure in the probability of setting a causality relation. Higher values for p lead to the inference of fewer causality relations among the tasks in the event log, and vice-versa.
    for i in range(len(alphabet) - 1):                                                                                  
        for j in range(1, len(alphabet)):                                                                               
            if ran.random() < pow(dependency_measure(alphabet[i], alphabet[j], log), influence_control):                
                cromossome[i][j] = 1                                                                                    
    for i in range(len(alphabet) - 1):                                                                                  # (eu mudei aqui do original, para quebrar em dois e evitar cálculo)
        if ran.random() < 0.5:                                                                                          
            cromossome[i][-1] = 1                                                                                       
    for i in range(1, len(alphabet)):                                                                                   # (eu mudei aqui do original, para quebrar em dois e evitar cálculo)
        if ran.random() < 0.5:                                                                                          
            cromossome[-1][i] = 1                                                                                       
    return cromossome                                                                                                   


def dependency_measure(t1, t2, log):                                                                                    #?
    dependency_measure = 0                                                                                              #?
    (l2l_t1_t2, follows_t1_t2) = d_m_measures(t1, t2, log)                                                              #?
    (l2l_t2_t1, follows_t2_t1) = d_m_measures(t2, t1, log)                                                              #?
    if (t1 == t2):                                                                                                      #?
        dependency_measure = (follows_t1_t2 / (follows_t1_t2 + 1))                                                      #?
    else:                                                                                                               #?
        if (t1 != t2):                                                                                                  #?
            if (l2l_t1_t2 == 0):                                                                                        #?
                dependency_measure = ((follows_t1_t2 - follows_t2_t1) / (follows_t1_t2 + follows_t2_t1 + 1))            #?
            else:                                                                                                       #?
                if (l2l_t1_t2 > 0):                                                                                     #?
                    dependency_measure = ((l2l_t1_t2 + l2l_t2_t1) / (l2l_t1_t2 + l2l_t2_t1 + 1))                        #?
                else:                                                                                                   #?
                    quit()                                                                                              #?
    return dependency_measure                                                                                           #?


def d_m_measures(t1, t2, log):                                                                                          #?
    l2l = 0 #the number of times that the substring "t1t2t1" occurs in a log.                                           #?
    follows = 0 #the number of times that a task is directly followed by another one. That is, how often the substring "t1t2" occurs in a log.     #?
    for i in range(len(log)):                                                                                           #?
        for j in range(len(log[i]) - 1):                                                                                #?
            if (log[i][j] == t1) and (log[i][j + 1] == t2):                                                             #?
                follows = follows + 1                                                                                   #?
                if (j + 2) <= (len(log[i]) - 1):                                                                        #?
                    if (log[i][j + 2] == t1):                                                                           #?
                        l2l = l2l + 1                                                                                   #?
    return l2l, follows                                                                                                 #?


def create_initial_individual2(cromossome, alphabet, reference_cromossome):
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
        if is_there_at_least_one_raw_active_task(cromossome, n) == False:
            chosen_task = ran.randint(1, len(cromossome) - 2)
            while reference_cromossome[n][chosen_task] == 0:
                chosen_task = ran.randint(1, len(cromossome) - 2)
            cromossome[n][chosen_task] = 1
    for n in range(1, len(cromossome) - 1):
        if is_there_at_least_one_column_active_task(cromossome, n) == False:
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
    number_of_produced_tokens = op.count_produced_tokens(cromossome)
    number_of_consumed_tokens = op.count_consumed_tokens(cromossome)
    if number_of_consumed_tokens > number_of_produced_tokens:
        op.increase_number_of_produced_tokens(cromossome, number_of_consumed_tokens)
    elif number_of_consumed_tokens < number_of_produced_tokens:
        op.increase_number_of_consumed_tokens(cromossome, number_of_consumed_tokens)

    return cromossome


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