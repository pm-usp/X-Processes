import petrinets as pn
import pm4py

def evaluate_population(population, alphabet, xes_log, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight):
    evaluation_sum = 0
    evaluation_values = []
    for i in range(len(population)):
        evaluation_values.append(evaluate_individual(population[i], alphabet, xes_log, i, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight))
        evaluation_sum = evaluation_sum + evaluation_values[i][0]
    return [evaluation_sum, evaluation_values]

def evaluate_individual(cromossome, alphabet, xes_log, i, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight):
    petrinet, initial_marking, final_marking = pn.create_petri_net(cromossome, alphabet)
    if algo_option == 'TOKEN_BASED-ETCONFORMANCE_TOKEN':
        fitness = pm4py.fitness_token_based_replay(xes_log, petrinet, initial_marking, final_marking)
        precision = pm4py.precision_token_based_replay(xes_log, petrinet, initial_marking, final_marking)
        generaliz = pm4py.generalization_tbr(xes_log, petrinet, initial_marking, final_marking)
        simplic = pm4py.simplicity_petri_net(petrinet, initial_marking, final_marking, variant='arc_degree')
        if (fitness['log_fitness'] == 0) or (precision == 0) or (generaliz == 0) or (simplic == 0):
            f_score = 0
        else:
            f_score = (fitness_weight + precision_weight + generalization_weight + simplicity_weight) / ((fitness_weight / fitness['log_fitness']) + (precision_weight / precision) + (generalization_weight / generaliz) + (simplicity_weight / simplic))
        return f_score, fitness['log_fitness'], precision, generaliz, simplic, i, 0                                             
    elif algo_option == 'ALIGNMENT_BASED-ETCONFORMANCE_TOKEN':
        fitness = pm4py.fitness_alignments(xes_log, petrinet, initial_marking, final_marking)
        precision = pm4py.precision_token_based_replay(xes_log, petrinet, initial_marking, final_marking)
        generaliz = pm4py.generalization_tbr(xes_log, petrinet, initial_marking, final_marking)
        simplic = pm4py.simplicity_petri_net(petrinet, initial_marking, final_marking, variant='arc_degree')
        if (fitness['log_fitness'] == 0) or (precision == 0) or (generaliz == 0) or (simplic == 0):
            f_score = 0
        else:
            f_score = (fitness_weight + precision_weight + generalization_weight + simplicity_weight) / ((fitness_weight / fitness['log_fitness']) + (precision_weight / precision) + (generalization_weight / generaliz) + (simplicity_weight / simplic))
        return f_score, fitness['log_fitness'], precision, generaliz, simplic, i, 0
    elif algo_option == 'ALIGNMENT_BASED-ALIGN_ETCONFORMANCE':
        fitness = pm4py.fitness_alignments(xes_log, petrinet, initial_marking, final_marking)
        precision = pm4py.precision_alignments(xes_log, petrinet, initial_marking, final_marking)
        generaliz = pm4py.generalization_tbr(xes_log, petrinet, initial_marking, final_marking)
        simplic = pm4py.simplicity_petri_net(petrinet, initial_marking, final_marking, variant='arc_degree')
        if (fitness['log_fitness'] == 0) or (precision == 0) or (generaliz == 0) or (simplic == 0):
            f_score = 0
        else:
            f_score = (fitness_weight + precision_weight + generalization_weight + simplicity_weight) / ((fitness_weight / fitness['log_fitness']) + (precision_weight / precision) + (generalization_weight / generaliz) + (simplicity_weight / simplic))
        return f_score, fitness['log_fitness'], precision, generaliz, simplic, i, 0