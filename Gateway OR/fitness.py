import petrinets as pn
import pm4py

def evaluate_population(population, alphabet, xes_log, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight, log_name, round, island, generation):
    evaluation_values = []
    evaluation_sum = 0
    for i, individual in enumerate(population):
        individual_evaluation = evaluate_individual(individual, alphabet, xes_log, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight, i, log_name, round, island, generation)
        evaluation_values.append(individual_evaluation)
        evaluation_sum += individual_evaluation[0]
    return [evaluation_sum, evaluation_values]

def evaluate_individual(cromossome, alphabet, xes_log, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight, i, log_name, round, island, generation):
    petrinet, initial_marking, final_marking = pn.create_petri_net(cromossome, alphabet, log_name, round, island, generation)
    metrics_functions = {
        'TOKEN_BASED-ETCONFORMANCE_TOKEN': (pm4py.fitness_token_based_replay, pm4py.precision_token_based_replay),
        'ALIGNMENT_BASED-ETCONFORMANCE_TOKEN': (pm4py.fitness_alignments, pm4py.precision_token_based_replay),
        'ALIGNMENT_BASED-ALIGN_ETCONFORMANCE': (pm4py.fitness_alignments, pm4py.precision_alignments)}
    fitness_fn, precision_fn = metrics_functions[algo_option]
    fitness = fitness_fn(xes_log, petrinet, initial_marking, final_marking)
    precision = precision_fn(xes_log, petrinet, initial_marking, final_marking)
    generaliz = pm4py.generalization_tbr(xes_log, petrinet, initial_marking, final_marking)
    simplic = pm4py.simplicity_petri_net(petrinet, initial_marking, final_marking, variant='arc_degree')
    if fitness['log_fitness'] == 0 or precision == 0 or generaliz == 0 or simplic == 0:
        return 0, fitness['log_fitness'], precision, generaliz, simplic, i, 0
    f_score = (fitness_weight + precision_weight + generalization_weight + simplicity_weight) / ((fitness_weight / fitness['log_fitness']) + (precision_weight / precision) + (generalization_weight / generaliz) + (simplicity_weight / simplic))
    return f_score, fitness['log_fitness'], precision, generaliz, simplic, i, 0