import petrinets as pn
import pm4py
import hashlib
import decorators
import log

@decorators.measure_time
def hash_cromossomo_fitness(cromossomo):
    cromossomo_str = str(cromossomo.tolist())
    return hashlib.sha256(cromossomo_str.encode()).hexdigest()

@decorators.measure_time
def hash_xes_log(xes_log):
    xes_log_str = str(xes_log)
    return hashlib.sha256(xes_log_str.encode()).hexdigest()

@decorators.measure_time
def evaluate_population(population, alphabet, fitness_weight, precision_weight, generalization_weight, simplicity_weight, input_log_name, round, island, generation, cache_fitness, cache_petri_net, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log):
    sampled_xes_log_file_path = 'input-logs/log-sampling/sampled-' + str(input_log_name.replace("\\", "-").replace("/", "-")).replace("input-logs-", "").replace(".xes", "").replace(".gz", "") + '.xes'
    cached_sampled_xes_log, last_file_size_sampled_xes_log = log.read_log_safely('sampled', sampled_xes_log_file_path, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log)
    xes_log = cached_sampled_xes_log
    evaluation_values = []
    evaluation_sum = 0
    for i, individual in enumerate(population):
        individual_evaluation = evaluate_individual(individual, alphabet, fitness_weight, precision_weight, generalization_weight, simplicity_weight, i, xes_log, input_log_name, round, island, generation, cache_fitness, cache_petri_net)
        evaluation_values.append(individual_evaluation)
        evaluation_sum += individual_evaluation[0]
    evaluated_population = [evaluation_sum, evaluation_values]
    return evaluated_population, cached_sampled_xes_log, cached_remaining_xes_log, last_file_size_sampled_xes_log, last_file_size_remaining_xes_log

@decorators.measure_time
def evaluate_individual(cromossome, alphabet, fitness_weight, precision_weight, generalization_weight, simplicity_weight, i, xes_log, input_log_name, round, island, generation, cache_fitness, cache_petri_net):
    cromossomo_hash = hash_cromossomo_fitness(cromossome)
    xes_log_hash = hash_xes_log(xes_log)
    cache_key = f"{cromossomo_hash}_{xes_log_hash}"
    cached_value = cache_fitness.get(cache_key)
    if isinstance(cached_value, tuple) and len(cached_value) == 4:
        fitness, precision, generaliz, simplic = cached_value
        if (isinstance(fitness, dict) and 'log_fitness' in fitness and isinstance(precision, (int, float)) and isinstance(generaliz, (int, float)) and isinstance(simplic, (int, float))):
            if fitness['log_fitness'] == 0 or precision == 0 or generaliz == 0 or simplic == 0:
                return 0, fitness['log_fitness'], precision, generaliz, simplic, i, 0
            f_score = (fitness_weight + precision_weight + generalization_weight + simplicity_weight) / ((fitness_weight / fitness['log_fitness']) + (precision_weight / precision) + (generalization_weight / generaliz) + (simplicity_weight / simplic))
            return f_score, fitness['log_fitness'], precision, generaliz, simplic, i, 0
    petrinet, initial_marking, final_marking = pn.create_petri_net(cromossome, alphabet, input_log_name, round, island, generation, cache_petri_net)
    fitness = pm4py.fitness_alignments(xes_log, petrinet, initial_marking, final_marking)
    precision = pm4py.precision_alignments(xes_log, petrinet, initial_marking, final_marking)
    generaliz = pm4py.generalization_tbr(xes_log, petrinet, initial_marking, final_marking)
    simplic = pm4py.simplicity_petri_net(petrinet, initial_marking, final_marking, variant='arc_degree')
    if (isinstance(fitness, dict) and 'log_fitness' in fitness and isinstance(precision, (int, float)) and isinstance(generaliz, (int, float)) and isinstance(simplic, (int, float))):
        if cache_key not in cache_fitness:
            cache_fitness[cache_key] = (fitness, precision, generaliz, simplic)
    if fitness['log_fitness'] == 0 or precision == 0 or generaliz == 0 or simplic == 0:
        return 0, fitness['log_fitness'], precision, generaliz, simplic, i, 0
    f_score = (fitness_weight + precision_weight + generalization_weight + simplicity_weight) / ((fitness_weight / fitness['log_fitness']) + (precision_weight / precision) + (generalization_weight / generaliz) + (simplicity_weight / simplic))
    return f_score, fitness['log_fitness'], precision, generaliz, simplic, i, 0