from pm4py.algo.analysis.woflan import algorithm as woflan
from pm4py.algo.conformance.alignments.petri_net import algorithm as alignments
from pm4py.algo.evaluation.replay_fitness import algorithm as replay_fitness
from pm4py.algo.evaluation.generalization import algorithm as calc_generaliz
from pm4py.algo.evaluation.precision import algorithm as calc_precision
from pm4py.algo.evaluation.simplicity import algorithm as calc_simplic
import petrinets as pn
import datetime as dt

def evaluate_population(population, alphabet, xes_log, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight):
    evaluation_sum = 0
    evaluation_values = []
    for i in range(len(population)):
        evaluation_values.append(evaluate_individual(population[i], alphabet, xes_log, i, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight))
        evaluation_sum = evaluation_sum + evaluation_values[i][0]
    return [evaluation_sum, evaluation_values]

def evaluate_individual(cromossome, alphabet, xes_log, i, algo_option, fitness_weight, precision_weight, generalization_weight, simplicity_weight):
    petrinet, initial_marking, final_marking = pn.create_petri_net(cromossome, alphabet)
    parameters = {replay_fitness.token_replay.Parameters.ACTIVITY_KEY: 'concept:name', replay_fitness.alignment_based.Parameters.ACTIVITY_KEY: 'concept:name', woflan.Parameters.RETURN_ASAP_WHEN_NOT_SOUND: True, woflan.Parameters.PRINT_DIAGNOSTICS: False, woflan.Parameters.RETURN_DIAGNOSTICS: False, alignments.Parameters.SHOW_PROGRESS_BAR: False, replay_fitness.token_replay.Parameters.SHOW_PROGRESS_BAR: False}
    if algo_option == 'TOKEN_BASED-ETCONFORMANCE_TOKEN':
        fitness = replay_fitness.apply(xes_log, petrinet, initial_marking, final_marking, parameters=parameters, variant=replay_fitness.Variants.TOKEN_BASED)
        precision = calc_precision.apply(xes_log, petrinet, initial_marking, final_marking, parameters=parameters, variant=calc_precision.Variants.ETCONFORMANCE_TOKEN)
        generaliz = calc_generaliz.apply(xes_log, petrinet, initial_marking, final_marking, parameters=parameters, variant=calc_generaliz.Variants.GENERALIZATION_TOKEN)
        simplic = calc_simplic.apply(petrinet)
        if (fitness['log_fitness'] == 0) or (precision == 0) or (generaliz == 0) or (simplic == 0):
            f_score = 0
        else:
            f_score = (fitness_weight + precision_weight + generalization_weight + simplicity_weight) / ((fitness_weight / fitness['log_fitness']) + (precision_weight / precision) + (generalization_weight / generaliz) + (simplicity_weight / simplic))
        return f_score, fitness['log_fitness'], precision, generaliz, simplic, i, 0                                             
    elif algo_option == 'ALIGNMENT_BASED-ETCONFORMANCE_TOKEN':
        fitness = replay_fitness.apply(xes_log, petrinet, initial_marking, final_marking, parameters=parameters, variant=replay_fitness.Variants.ALIGNMENT_BASED)
        precision = calc_precision.apply(xes_log, petrinet, initial_marking, final_marking, parameters=parameters, variant=calc_precision.Variants.ETCONFORMANCE_TOKEN)
        generaliz = calc_generaliz.apply(xes_log, petrinet, initial_marking, final_marking, parameters=parameters, variant=calc_generaliz.Variants.GENERALIZATION_TOKEN)
        simplic = calc_simplic.apply(petrinet)                                                                          
        if (fitness['averageFitness'] == 0) or (precision == 0) or (generaliz == 0) or (simplic == 0):
            f_score = 0
        else:
            f_score = (fitness_weight + precision_weight + generalization_weight + simplicity_weight) / ((fitness_weight / fitness['averageFitness']) + (precision_weight / precision) + (generalization_weight / generaliz) + (simplicity_weight / simplic))
        return f_score, fitness['averageFitness'], precision, generaliz, simplic, i, 0
    elif algo_option == 'ALIGNMENT_BASED-ALIGN_ETCONFORMANCE':
        fitness = replay_fitness.apply(xes_log, petrinet, initial_marking, final_marking, parameters=parameters, variant=replay_fitness.Variants.ALIGNMENT_BASED)
        precision = calc_precision.apply(xes_log, petrinet, initial_marking, final_marking, parameters=parameters, variant=calc_precision.Variants.ALIGN_ETCONFORMANCE)
        generaliz = calc_generaliz.apply(xes_log, petrinet, initial_marking, final_marking, parameters=parameters, variant=calc_generaliz.Variants.GENERALIZATION_TOKEN)
        simplic = calc_simplic.apply(petrinet)                                                                          
        if (fitness['averageFitness'] == 0) or (precision == 0) or (generaliz == 0) or (simplic == 0):
            f_score = 0
        else:
            f_score = (fitness_weight + precision_weight + generalization_weight + simplicity_weight) / ((fitness_weight / fitness['averageFitness']) + (precision_weight / precision) + (generalization_weight / generaliz) + (simplicity_weight / simplic))
        return f_score, fitness['averageFitness'], precision, generaliz, simplic, i, 0